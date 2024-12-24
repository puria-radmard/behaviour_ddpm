from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Dict, Optional

from abc import ABC



try:
    from sampling_ddpm.ddpm.model.time_repr import *
    from sampling_ddpm.ddpm.model.residual import *
    from sampling_ddpm.ddpm.model.input import *
except ImportError:
    from ddpm.model.time_repr import *
    from ddpm.model.residual import *
    from ddpm.model.input import *




class DDPMReverseProcess(nn.Module, ABC):
    """
    For now just the simplest case where we have a single time varying sigma schedule...

    sigma2xt_schedule is beta_t in the original literature, of shape [T]
        sigma2xt_schedule[i] gives \sigma^2_{i+1} e.g. sigma2xt_schedule[0] gives \sigma^2_1 etc.
    """

    def __init__(
        self,
        hidden_size: int,
        residual_model: ResidualModel,
        input_model: InputModelBlock,
        sigma2xt_schedule: _T,
        time_embedding_size: int,
        sample_space_size: int = 2,
        euler_alpha: float = 0.1,
    ) -> None:
        
        super().__init__()

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
        
        self.T = len(sigma2xt_schedule)

        self.residual_model = residual_model
        # self.time_embeddings = nn.Embedding(self.T, time_embedding_size)
        self.time_embeddings = TimeEmbeddingBlock(self.T, time_embedding_size)
        self.input_model = input_model
        assert input_model.output_size == residual_model.input_size
        
        self.euler_alpha = euler_alpha
        self.sample_space_size = sample_space_size
        self.hidden_size = hidden_size

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
        return super(DDPMReverseProcess, self).to(*args, **kwargs)

    @property
    def behaviour_projection_matrix(self,) -> _T:
        return self.linking_matrix.T    # [dim x, dim y]

    def noise(self, y_samples: _T) -> Dict[str, _T]:
        """
        y_samples of shape [..., dim y]

        Both outputs of shape [..., T, dim x]
        """
        extra_dims = len(y_samples.shape[:-1])
        
        # [..., 1, dim x]
        x_0 = (y_samples @ self.behaviour_projection_matrix.T).unsqueeze(-2)
        x_0 = x_0.repeat(*[1]*extra_dims, self.T, 1)
        epsilon = torch.randn_like(x_0)

        # [..., T, dim x]
        x_t_means = self.a_t_schedule[*[None]*extra_dims].unsqueeze(-1) * x_0 
        x_t = x_t_means + self.root_b_t_schedule[*[None]*extra_dims].unsqueeze(-1) * epsilon

        return {
            'x_t': x_t,
            'epsilon': epsilon
        }
    
    def residual(self, x_samples: _T, network_input: _T, epsilon: Optional[_T] = None, **kwargs_for_residual_model) -> Dict[str, _T]:
        """
        x_samples of shape [..., T, dim x]
        input_vector of shape [..., <shape Z>]
        if epsilon provided, of shape [..., T, dim x]
        """
        input_vector = self.input_model(network_input)

        assert tuple(x_samples.shape[-2:]) == (self.T, self.hidden_size), f"Expected x_samples shape to end with {(self.T, self.sample_space_size)} but got {tuple(x_samples.shape[-2:])}"
        assert input_vector.shape[-1] == (self.residual_model.input_size), f"Expected input_vector shape to end with {(self.T, self.residual_model.input_size)} but got {tuple(input_vector.shape[-2:])}"
        
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
    
    def generate_samples(self, network_input: _T, samples_shape: Optional[int] = None, base_samples: Optional[_T] = None, turn_off_noise: bool = False) -> Dict[str, _T]:
        """
        input_vector of shape [..., <shape Z>]
        If provided, base_samples of shape [..., dim x]
        Otherwise, give samples_shape = [...]
        """
        assert (samples_shape is None) != (base_samples is None)

        input_vector = self.input_model(network_input)

        if base_samples is None:
            base_samples = torch.randn(*samples_shape, self.hidden_size, device = self.sigma2xt_schedule.device) * self.base_std
        else:
            samples_shape = base_samples.shape[:-1]
            assert base_samples.shape[-1] == self.hidden_size

        sample_trajectory = []
        noises = torch.randn(*samples_shape, self.T, self.hidden_size, device = base_samples.device)
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
        behaviour_samples = base_samples.squeeze(-2) @ self.linking_matrix.T

        return {
            'sample_trajectory': sample_trajectory,
            'behaviour_samples': behaviour_samples
        }
            
            
