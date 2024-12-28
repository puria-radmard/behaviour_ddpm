from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Dict, Optional

from abc import ABC

from ddpm.tasks.sample import WhiteNoiseStripExampleSampleGenerator

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
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device = 'cuda'
    ) -> None:
        super().__init__()

        self.sample_shape = sample_shape

        self.T = len(sigma2xt_schedule)

        self.residual_model = residual_model
        # self.time_embeddings = nn.Embedding(self.T, time_embedding_size)
        self.time_embeddings = TimeEmbeddingBlock(self.T, time_embedding_size, device=device)
        self.input_model = input_model
        assert input_model.network_input_size == residual_model.input_size,\
            f"Got an input model (sensory -> residual network) with output size {input_model.network_input_size}, but a residual model that accepts inputs of size {residual_model.input_size}"

        self.sigma2xt_schedule = sigma2xt_schedule          # t = 1, ..., T
        assert len(sigma2xt_schedule.shape) == 1
        self.t_schedule = torch.arange(self.T)   # Indexes self.time_embeddings.time_embs
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
        assert list(x.shape[-len(self.sample_shape):]) == self.sample_shape,\
            f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x.shape}"

    def noise(self, x_0: _T) -> Dict[str, _T]:
        """
        x_0 of shape [batch, <shape x>]

        Both outputs of shape [..., T, dim x]
        """
        self.verify_samples(x_0)

        x_0 = x_0.unsqueeze(1).repeat(1, self.T, *[1]*len(self.sample_shape))
        epsilon = torch.randn_like(x_0)

        # [..., T, dim x]
        x_t_means = self.a_t_schedule[None, :, *[None]*len(self.sample_shape)] * x_0 
        x_t = x_t_means + self.root_b_t_schedule[None, :, *[None]*len(self.sample_shape)] * epsilon

        return {
            'x_t': x_t,
            'epsilon': epsilon
        }

    def residual(self, x_samples: _T, network_input: _T, **kwargs_for_residual_model) -> Dict[str, _T]:  # epsilon: Optional[_T] = None, 
        """
        x_samples of shape [B, T, <shape x>]
        network_input of shape [<shape Z>]       --> same for all samples and timesteps in this class!
        if epsilon provided, of shape [B, T, dim x]
        """
        batch_size = x_samples.shape[0]
        input_vector: _T = self.input_model(network_input)

        assert tuple(input_vector.shape) == (self.residual_model.input_size,),\
            f"Expected input_vector shape to be just {(self.residual_model.input_size)} but got {tuple(input_vector.shape)}"

        assert tuple(x_samples.shape) == (batch_size, self.T, *self.sample_shape),\
            f"Expected x_samples shape to be {(batch_size, self.T, self.sample_space_size)} but got {x_samples.shape}"

        input_vector = input_vector.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.T, 1)
        t_embeddings = self.time_embeddings(self.t_schedule)
        epsilon_hat = self.residual_model(x = x_samples, t_embeddings_schedule = t_embeddings, input_vector = input_vector, **kwargs_for_residual_model)    # [B, T, <sample shape>]
        assert x_samples.shape == epsilon_hat.shape
        
        # if epsilon is not None:
        #     assert epsilon.shape == x_samples.shape
        #     mse = torch.square(epsilon_hat - epsilon).reshape(batch_size, self.T, -1).mean(-1)   # [..., T]
        #     # scaled_mse = self.mse_scaler_schedule[*[None]*extra_dims] * mse[...,1:] # [..., T-1]

        #     ret['mse'] = mse

        return epsilon_hat

    @torch.no_grad()
    def generate_samples(self, network_input: _T, num_samples: Optional[int] = None, base_samples: Optional[_T] = None, turn_off_noise: bool = False) -> Dict[str, _T]:
        """
        input_vector of shape [B, <shape Z>]
        If provided, base_samples of shape [B, <shape x>]
        Otherwise, give B = num_samples

        sample_trajectory of shape [B, T, <shape x>]
        samples of shape [B, <shape x>]
        early_x0_preds of shape [B, T, <shape x>]
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
        
        noises = torch.randn(num_samples, self.T, *self.sample_shape, device = base_samples.device)
        base_samples = base_samples.unsqueeze(1)
        t_embeddings = self.time_embeddings(self.t_schedule)
        
        if turn_off_noise:
            noises *= 0.0

        sample_trajectory = []
        early_x0_preds = []

        for t_idx in range(1, self.T + 1):

            t_embedding = t_embeddings[-t_idx][None]

            predicted_residual = self.residual_model(base_samples, t_embedding, input_vector)
            # true_x0 = input_vector[...,2 * input_vector[:,:,-3:].argmax(-1)[0]: 2 * (input_vector[:,:,-3:].argmax(-1) + 1)[0]]
            # true_x0 = 0.5 * WhiteNoiseStripExampleSampleGenerator.generate_strip_image(32, 6, torch.arctan2(input_vector[:,0,1].T, input_vector[:,0,0].T)).unsqueeze(1)
            # predicted_residual = (base_samples - (self.a_t_schedule[-t_idx] * true_x0)) / self.root_b_t_schedule[-t_idx]

            # noise_scaler_schedule = root_b_t_schedule[:-1] / root_b_t_schedule[1:] * torch.sqrt(sigma2xt_schedule[1:])        # t = 2, ..., T
            # noise_scaler_schedule = torch.concat([torch.zeros(1), noise_scaler_schedule], 0)                                  # Still t = 2, ..., T but with a zero appended to the front for ease!
            scaled_noise = noises[:,[-t_idx],...] * self.noise_scaler_schedule[-t_idx]

            # base_samples_scaler_schedule = 1.0 / torch.sqrt(1.0 - sigma2xt_schedule)                                          # t = 1, ..., T
            scaled_base_samples = base_samples * self.base_samples_scaler_schedule[-t_idx]

            # residual_scaler_schedule = sigma2xt_schedule / torch.sqrt(1.0 - sigma2xt_schedule) / root_b_t_schedule
            scaled_residual = self.residual_scaler_schedule[-t_idx] * predicted_residual # [B, 1, dim x]

            # x_t = (a_t_schedule * x_0) + root_b_t_schedule * epsilon
            early_x0_pred = (base_samples - self.root_b_t_schedule[-t_idx] * predicted_residual) / self.a_t_schedule[-t_idx]

            base_samples = scaled_base_samples - scaled_residual + scaled_noise

            sample_trajectory.append(base_samples.detach().cpu())
            early_x0_preds.append(early_x0_pred.detach().cpu())
        
        sample_trajectory = torch.concat(sample_trajectory, 1) # [B, T, <shape x>]
        early_x0_preds = torch.concat(early_x0_preds, 1) # [B, T, <shape x>]

        return {
            'sample_trajectory': sample_trajectory,
            'samples': base_samples.squeeze(1),
            'early_x0_preds': early_x0_preds,
        }
            
            



