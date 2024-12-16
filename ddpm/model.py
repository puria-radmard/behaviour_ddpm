from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T
import torch.nn.functional as F

import numpy as np

from typing import List, Dict, Optional, Any

from abc import ABC, abstractmethod


class ResidualModel(nn.Module):
    """
    ϵ_θ (x_t, t, Z)
    Just a linear-relu cascade but with an extra set of dim for time and inputs at each layer
    """
    def __init__(self, state_space_size: int, recurrence_hidden_layers: List[int], input_size: int) -> None:
        super().__init__()
        
        self.state_space_size = state_space_size
        self.recurrence_hidden_layers = recurrence_hidden_layers
        self.input_size = input_size
        all_layer_sizes = recurrence_hidden_layers + [state_space_size]
        recurrence_layers = [nn.Linear(state_space_size + 1 + input_size, all_layer_sizes[0]), nn.ReLU()]        # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            # recurrence_layers.extend([nn.Linear(all_layer_sizes[i], op_s), nn.ReLU()])
            recurrence_layers.extend([nn.Linear(all_layer_sizes[i] + 1 + input_size, op_s), nn.ReLU()])
        recurrence_layers = recurrence_layers[:-1]
        self.layers = nn.ModuleList(recurrence_layers)  # R^N -> R^N

    @staticmethod
    def concatenate_with_time_and_input(x: _T, t_schedule: _T, input_vector: _T) -> _T:
        reshaped_t_schedule = t_schedule.reshape(*[1 for _ in range(len(x.shape) - 2)], -1, 1).repeat(*x.shape[:-2], 1, 1)
        input_vector_over_time = input_vector.unsqueeze(-2).repeat(*[1]*(len(input_vector.shape)-1), t_schedule.shape[0], 1)
        x_concat = torch.concat([x, reshaped_t_schedule.to(x.device, x.dtype), input_vector_over_time.to(x.device, x.dtype)], -1)
        return x_concat

    def forward(self, x: _T, t_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_schedule of shape [T]
        input_vector of shape [...], passed to all

        x[...,i,:] is x_{t_schedule[i]}
            e.g. 
                if t_schedule[i] = 1, then x[...,i,:] is x_1
                if t_schedule[i] = 5, then x[...,i,:] is x_5
            But actually, the numerical values of t_schedule won't be the integer values printed here - see below
        """
        # x_concat = self.concatenate_with_time(x, t_schedule)
        # return self.model(x_concat) # sequential
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x_concat = self.concatenate_with_time_and_input(x, t_schedule, input_vector)
                x = layer(x_concat)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
            else:
                raise NotImplementedError
        return x
    


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
        sigma2xt_schedule: _T,
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

        self.residual_model = residual_model
        
        self.euler_alpha = euler_alpha
        self.sample_space_size = sample_space_size
        self.hidden_size = hidden_size

        self.sigma2xt_schedule = sigma2xt_schedule          # t = 1, ..., T
        assert len(sigma2xt_schedule.shape) == 1
        self.T = len(sigma2xt_schedule)
        self.t_schedule = torch.linspace(0., 1., self.T)   # Just to normalise! Here, self.t_schedule = 0 when t = 1
        self.a_t_schedule = torch.cumprod(torch.sqrt(1.0 - self.sigma2xt_schedule), 0)  # t = 1, ..., T
        self.root_b_t_schedule = torch.sqrt(1.0 - torch.square(self.a_t_schedule))      # t = 1, ..., T

        # Generation
        self.base_std = self.root_b_t_schedule[-1]  # scalar, final variance = b_T
        noise_scaler_schedule = torch.square(self.root_b_t_schedule[:-1] / self.root_b_t_schedule[1:])    # t = 2, ..., T
        self.noise_scaler_schedule = torch.concat([torch.zeros(1, device = noise_scaler_schedule.device), noise_scaler_schedule], 0)    # t = 2, ..., T but with a zero appended to the front for ease!
        self.base_samples_scaler_schedule = 1.0 / torch.sqrt(1.0 - self.sigma2xt_schedule)                  # t = 1, ..., T
        self.residual_scaler_schedule = torch.sqrt(self.sigma2xt_schedule) / torch.sqrt(1.0 - self.sigma2xt_schedule) / self.root_b_t_schedule
        
        # Training
        mse_scaler_schedule_num = (1 - self.a_t_schedule[1:] ** 2.0) * self.sigma2xt_schedule[1:]    # t = 2, ..., T
        mse_scaler_schedule_denom = 2 * (1 - self.a_t_schedule[:-1] ** 2.0) * (1.0 - self.sigma2xt_schedule[1:])    # t = 2, ..., T
        self.mse_scaler_schedule = mse_scaler_schedule_num / mse_scaler_schedule_denom

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
    
    def residual(self, x_samples: _T, input_vector: _T, epsilon: Optional[_T] = None) -> Dict[str, _T]:
        """
        x_samples of shape [..., T, dim x]
        input_vector of shape [..., dim Z]
        if epsilon provided, of shape [..., T, dim x]
        """
        extra_dims = len(x_samples.shape[:-2])

        assert tuple(x_samples.shape[-2:]) == (self.T, self.hidden_size), f"Expected x_samples shape to end with {(self.T, self.sample_space_size)} but got {tuple(x_samples.shape[-2:])}"
        assert input_vector.shape[-1] == (self.residual_model.input_size), f"Expected input_vector shape to end with {(self.T, self.residual_model.input_size)} but got {tuple(input_vector.shape[-2:])}"

        epsilon_hat = self.residual_model(x_samples, self.t_schedule, input_vector)
        assert x_samples.shape == epsilon_hat.shape

        ret = {'epsilon_hat': epsilon_hat}
        
        if epsilon is not None:
            assert epsilon.shape == x_samples.shape
            mse = torch.square(epsilon_hat - epsilon).sum(-1)   # [..., T]
            # scaled_mse = self.mse_scaler_schedule[*[None]*extra_dims] * mse[...,1:] # [..., T-1]
            scaled_mse = mse

            ret['scaled_mse'] = scaled_mse
        
        return ret
    
    def generate_samples(self, input_vector: _T, samples_shape: Optional[int] = None, base_samples: Optional[_T] = None, turn_off_noise: bool = False) -> Dict[str, _T]:
        """
        input_vector of shape [..., dim Z]
        If provided, base_samples of shape [..., dim x]
        Otherwise, give samples_shape = [...]
        """
        assert (samples_shape is None) != (base_samples is None)

        if base_samples is None:
            base_samples = torch.randn(*samples_shape, self.hidden_size, device = self.sigma2xt_schedule.device) * (self.base_std ** 0.5)
        else:
            samples_shape = base_samples.shape[:-1]
            assert base_samples.shape[-1] == self.hidden_size

        sample_trajectory = []
        noises = torch.randn(*samples_shape, self.T, self.hidden_size, device = base_samples.device)
        if turn_off_noise:
            noises *= 0.0

        base_samples = base_samples.unsqueeze(-2)

        for t_idx in range(1, self.T + 1):

            t_tilde = self.t_schedule[-t_idx][None]
            noise = noises[...,[-t_idx],:]
            base_samples_scaler = self.base_samples_scaler_schedule[-t_idx]
            noise_scaler = self.noise_scaler_schedule[-t_idx]
            residual_scaler = self.residual_scaler_schedule[-t_idx]

            new_residual = self.residual_model(base_samples, t_tilde, input_vector) # [..., 1, dim x]
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
            
            
