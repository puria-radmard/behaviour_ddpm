from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Dict, Optional

from abc import ABC



from ddpm.model.time_repr import *
from ddpm.model.residual import *
from ddpm.model.input import *



class DDPMReverseProcessBase(nn.Module, ABC):
    """
    Time varying sigma schedule:
        sigma2xt_schedule is beta_t in the original literature, of shape [T]
        sigma2xt_schedule[i] gives \sigma^2_{i+1} e.g. sigma2xt_schedule[0] gives \sigma^2_1 etc.
    """

    def __init__(
        self,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: ResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
    ) -> None:
        super().__init__()

        self.sample_shape = sample_shape

        self.T = len(sigma2xt_schedule)

        self.residual_model = residual_model
        # self.time_embeddings = nn.Embedding(self.T, time_embedding_size)
        self.time_embeddings = TimeEmbeddingBlock(self.T, time_embedding_size)
        self.input_model = input_model
        assert input_model.output_shape == residual_model.output_shape

        self.sigma2xt_schedule = sigma2xt_schedule          # t = 1, ..., T
        assert len(sigma2xt_schedule.shape) == 1
        self.t_schedule = torch.arange(self.T)   # Indexes nn.Embedding
        self.a_t_schedule = torch.cumprod(torch.sqrt(1.0 - self.sigma2xt_schedule), 0)  # t = 1, ..., T
        self.root_b_t_schedule = torch.sqrt(1.0 - torch.square(self.a_t_schedule))      # t = 1, ..., T

        # Generation
        self.base_std = self.root_b_t_schedule[-1]  # scalar, final variance = b_T
        noise_scaler_schedule = self.root_b_t_schedule[:-1] / self.root_b_t_schedule[1:] * torch.sqrt(self.sigma2xt_schedule[1:])    # t = 2, ..., T
        self.noise_scaler_schedule = torch.concat([torch.zeros(1, device = noise_scaler_schedule.device), noise_scaler_schedule], 0)    # Still t = 2, ..., T but with a zero appended to the front for ease!
        self.base_samples_scaler_schedule = 1.0 / torch.sqrt(1.0 - self.sigma2xt_schedule)                  # t = 1, ..., T
        self.residual_scaler_schedule = self.sigma2xt_schedule / torch.sqrt(1.0 - self.sigma2xt_schedule) / self.root_b_t_schedule
        
        ## Training
        #mse_scaler_schedule_num = (1 - self.a_t_schedule[1:] ** 2.0) * self.sigma2xt_schedule[1:]    # t = 2, ..., T
        #mse_scaler_schedule_denom = 2 * (1 - self.a_t_schedule[:-1] ** 2.0) * (1.0 - self.sigma2xt_schedule[1:])    # t = 2, ..., T
        #self.mse_scaler_schedule = mse_scaler_schedule_num / mse_scaler_schedule_denom

    def to(self, *args, **kwargs):
        self.sigma2xt_schedule = self.sigma2xt_schedule.to(*args, **kwargs)
        self.t_schedule = self.t_schedule.to(*args, **kwargs)
        self.a_t_schedule = self.a_t_schedule.to(*args, **kwargs)
        self.root_b_t_schedule = self.root_b_t_schedule.to(*args, **kwargs)
        # self.mse_scaler_schedule = self.mse_scaler_schedule.to(*args, **kwargs)
        self.noise_scaler_schedule = self.noise_scaler_schedule.to(*args, **kwargs)
        return super(DDPMReverseProcessBase, self).to(*args, **kwargs)

    def verify_samples(self, x: _T):
        assert list(x.shape[-len(self.sample_shape):]) == self.sample_shape

    def noise(self, x_0: _T) -> Dict[str, _T]:
        """
        x_0 of shape [batch, dim x]

        Both outputs of shape [..., T, dim x]
        """
        self.verify_samples(x_0)

        x_0 = x_0.repeat(1, self.T, 1)
        epsilon = torch.randn_like(x_0)

        # [..., T, dim x]
        x_t_means = self.a_t_schedule.unsqueeze(0).unsqueeze(-1) * x_0 
        x_t = x_t_means + self.root_b_t_schedule.unsqueeze(0).unsqueeze(-1) * epsilon

        return {
            'x_t': x_t,
            'epsilon': epsilon
        }

    def residual(self, x_samples: _T, network_input: _T, epsilon: Optional[_T] = None, **kwargs_for_residual_model) -> Dict[str, _T]:
        """
        x_samples of shape [B, T, <shape x>]
        network_input of shape [<shape Z>]       --> same for all samples and timesteps in this class!
        if epsilon provided, of shape [B, T, dim x]
        """
        batch_size = x_samples.shape[0]
        input_vector: _T = self.input_model(network_input)

        assert tuple(input_vector.shape) == (self.residual_model.input_size,),\
            f"Expected input_vector shape to be just {(self.residual_model.input_size)} but got {tuple(input_vector.shape)}"

        assert tuple(x_samples.shape) == tuple(epsilon.shape) == (batch_size, self.T, *self.sample_shape),\
            f"Expected x_samples shape to be {(batch_size, self.T, self.sample_space_size)} but got {x_samples.shape}"

        input_vector = input_vector.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.T, 1)
        
        t_embeddings = self.time_embeddings(self.t_schedule)
        epsilon_hat = self.residual_model(x = x_samples, t_embeddings_schedule = t_embeddings, input_vector = input_vector, **kwargs_for_residual_model)
        assert x_samples.shape == epsilon_hat.shape

        ret = {'epsilon_hat': epsilon_hat}
        
        if epsilon is not None:
            assert epsilon.shape == x_samples.shape
            mse = torch.square(epsilon_hat - epsilon).sum(-1)   # [..., T]
            # scaled_mse = self.mse_scaler_schedule[*[None]*extra_dims] * mse[...,1:] # [..., T-1]

            ret['mse'] = mse
        
        return ret

    
    def generate_samples(self, network_input: _T, num_samples: Optional[int] = None, base_samples: Optional[_T] = None, turn_off_noise: bool = False) -> Dict[str, _T]:
        """
        input_vector of shape [B, <shape Z>]
        If provided, base_samples of shape [B, <shape x>]
        Otherwise, give B = num_samples
        """
        assert (num_samples is None) != (base_samples is None)

        if base_samples is None:
            base_samples = torch.randn(num_samples, *self.sample_shape, device = self.sigma2xt_schedule.device) * self.base_std
        else:
            num_samples = base_samples.shape[0]
            assert tuple(base_samples.shape) == (num_samples, *self.sample_shape)

        input_vector = self.input_model(network_input)
        assert tuple(input_vector.shape) == (self.residual_model.input_size,),\
            f"Expected input_vector shape to be just {(self.residual_model.input_size)} but got {tuple(input_vector.shape)}"
        input_vector = input_vector.unsqueeze(0).unsqueeze(0).repeat(num_samples, 1, 1)
        
        sample_trajectory = []
        noises = torch.randn(num_samples, self.T, self.hidden_size, device = base_samples.device)
        if turn_off_noise:
            noises *= 0.0

        base_samples = base_samples.unsqueeze(-2)

        t_embeddings = self.time_embeddings(self.t_schedule)

        for t_idx in range(1, self.T + 1):

            t_embedding = t_embeddings[-t_idx][None]
            noise = noises[...,[-t_idx],:]
            base_samples_scaler = self.base_samples_scaler_schedule[-t_idx]
            noise_scaler = self.noise_scaler_schedule[-t_idx]
            residual_scaler = self.residual_scaler_schedule[-t_idx]

            new_residual = self.residual_model(base_samples, t_embedding, input_vector) # [..., 1, dim x]
            scaled_residual = residual_scaler * new_residual
            scaled_mean = base_samples / base_samples_scaler
            scaled_noise = noise * noise_scaler

            base_samples = scaled_mean - scaled_residual + scaled_noise

            sample_trajectory.append(base_samples.cpu().detach())
        
        sample_trajectory = torch.concat(sample_trajectory, -2) # [..., T, dim x]

        return {
            'sample_trajectory': sample_trajectory,
            'samples': base_samples.squeeze(-2)
        }
            
            





class EmbeddedTabularDDPMReverseProcess(DDPMReverseProcessBase):
    """
    Samples from a 2D space, diffusion in a higher dimensional space
    """

    def __init__(
        self,
        hidden_size: int,
        residual_model: ResidualModel,
        input_model: InputModelBlock,
        sigma2xt_schedule: _T,
        time_embedding_size: int,
        sample_space_size: int = 2,
    ) -> None:
        
        super().__init__(sigma2xt_schedule, residual_model, input_model, time_embedding_size, [sample_space_size])

        if hidden_size == sample_space_size:
            self.register_buffer('linking_matrix', torch.eye(hidden_size), persistent=True)   # [dim y, dim x]
            self.register_buffer('linking_matrix_kernel', torch.tensor([]), persistent=True)   # []
            self.register_buffer('rotation_matrix', torch.eye(hidden_size), persistent=True)   # []
        elif hidden_size > sample_space_size:
            gaus = torch.randn(hidden_size, hidden_size)
            svd = torch.linalg.svd(gaus)
            orth = svd[0] @ svd[2]
            self.register_buffer('linking_matrix', orth[:sample_space_size], persistent=True)   # [dim y, dim x]
            self.register_buffer('linking_matrix_kernel', orth[sample_space_size:], persistent=True)   # [dim x - dim y, dim x]
            self.register_buffer('rotation_matrix', orth, persistent=True)   # [dim x, dim x]
        else:
            raise NotImplementedError
        
        self.sample_space_size = sample_space_size
        self.hidden_size = hidden_size

    @property
    def behaviour_projection_matrix(self,) -> _T:
        return self.linking_matrix.T    # [dim x, dim y]

    def noise(self, y_samples: _T) -> Dict[str, _T]:
        """
        y_samples of shape [batch, dim y]
        """
        # [..., 1, dim x]
        x_0 = (y_samples @ self.behaviour_projection_matrix.T).unsqueeze(-2)
        return super(EmbeddedTabularDDPMReverseProcess, self).noise(x_0)

    def generate_samples(self, network_input: _T, num_samples: Optional[int] = None, base_samples: Optional[_T] = None, turn_off_noise: bool = False) -> Dict[str, _T]:
        """
        Have to make a slight change here!

        Because self.sample_shape = [2], but the actual diffusion is done in higher dimensions, we have to 
        """
        samples_dict = super().generate_samples(network_input=network_input, num_samples=num_samples, base_samples=base_samples, turn_off_noise=turn_off_noise)
        samples_dict['final_hidden_samples'] = samples_dict.pop('samples')
        samples_dict['samples'] = samples_dict['final_hidden_samples'] @ self.linking_matrix.T


        return samples_dict
