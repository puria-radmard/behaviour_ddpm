from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T
import torch.nn.functional as F

import numpy as np

from typing import List, Dict, Optional, Any

from abc import ABC, abstractmethod



class RecurrenceModel(nn.Module):
    "just a linear-relu cascade but with an extra set of dim for time and inputs at each layer"

    def __init__(self, state_space_size: int, recurrence_hidden_layers: List[int], input_size: int) -> None:
        super().__init__()
        self.state_space_size = state_space_size
        self.recurrence_hidden_layers = recurrence_hidden_layers
        all_layer_sizes = recurrence_hidden_layers + [state_space_size]
        recurrence_layers = [nn.Linear(state_space_size + 1 + input_size, all_layer_sizes[0]), nn.ReLU()]        # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            # recurrence_layers.extend([nn.Linear(all_layer_sizes[i], op_s), nn.ReLU()])
            recurrence_layers.extend([nn.Linear(all_layer_sizes[i] + 1 + input_size, op_s), nn.ReLU()])
        recurrence_layers = recurrence_layers[:-1]
        # self.model = nn.Sequential(*recurrence_layers)   # R^N -> R^N_+
        self.layers = nn.ModuleList(recurrence_layers)
        self.input_size = input_size

    @staticmethod
    def concatenate_with_time_and_input(x: _T, t_schedule: _T, input_vector: _T) -> _T:
        reshaped_t_schedule = t_schedule.reshape(*[1 for _ in range(len(x.shape) - 2)], -1, 1).repeat(*x.shape[:-2], 1, 1)
        input_vector_over_time = input_vector.unsqueeze(-2).repeat(*[1]*(len(input_vector.shape)-1), t_schedule.shape[0], 1)
        x_concat = torch.concat([x, reshaped_t_schedule.to(x.dtype), input_vector_over_time.to(x.dtype)], -1)
        return x_concat

    def forward(self, x: _T, t_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_schedule of shape [T]
        input_vector of shape [...]
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




class DDPMReverseProcessBase(nn.Module, ABC):
    """
    Different subclasses treat the time schedule differently - see child docstrings
    """
    
    recon_epsilon: float = 0.0025
    
    def __init__(
        self,
        hidden_size: int,
        recurrence_model: RecurrenceModel,
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

        self.recurrence_model = recurrence_model
        
        self.euler_alpha = euler_alpha
        self.sample_space_size = sample_space_size
        self.hidden_size = hidden_size

    @staticmethod
    def unsqueeze_for_arbitrary_front_dims(tensor: _T, extra_dims: List[int]) -> _T:
        return tensor[*[None]*len(extra_dims)].repeat(*extra_dims, *[1]*len(tensor.shape))

    @property
    def behvaiour_projection_matrix(self,) -> _T:
        return self.linking_matrix.T    # [dim x, dim y]

    def one_step_reverse_generative_process(self, x_samples: _T, **kwargs) -> Dict[str, _T]:
        """
        One shot forward disitribution prediction

        x_samples is output of run_forward_process, shaped [..., T, dim_x]
            x_samples[..., 1:, :] used to generate one step predictions, which are kept in that order
            x_samples[..., 0, :] used to generate decoding predictions

        Kwargs likely includes {"input_vector" [...]}

        outputs:
            x_mu_ps [..., T-1, nh]
            x_Sigma_chol_ps [..., T-1, nh, nh], lower-left ---> diagonal now
            y_mu_p [..., dim_y (2)]
            y_Sigma_p [..., dim_y, dim_y]
        
        TODO: MAKE y_Sigma_p [...] SINCE IT'S JUST DIAGONAL - ALSO FIX THE CROSS ENTROPY FUNCTION
        """
        assert list(x_samples.shape[-2:]) == [self.num_hidden_steps, self.hidden_size], f"x_samples of shape {x_samples.shape} don't match {[self.num_hidden_steps, self.hidden_size]} in last two dimensions"

        sigmaxt_schedule = self.sigma2xt_schedule.unsqueeze(-1).unsqueeze(-1) ** 0.5

        # [..., T, dim_x] (first one later removed) and [..., T-1, dim_x, dim_x]
        one_step_decoding_means = ((1 - self.euler_alpha) * x_samples) + (self.euler_alpha * self.recurrence_model(x_samples, self.recurrence_t_schedule, **kwargs))
        one_step_decoding_sigma2_chols = self.unsqueeze_for_arbitrary_front_dims(
            torch.eye(self.hidden_size).unsqueeze(0).to(x_samples.device) * sigmaxt_schedule,
            x_samples.shape[:-2]
        )

        # [..., dim_y = 2] and [..., dim_y, dim_y]
        reconstruction_mean = one_step_decoding_means[...,0,:] @ self.behvaiour_projection_matrix
        reconstruction_sigma2 = self.unsqueeze_for_arbitrary_front_dims(torch.eye(self.sample_space_size) * self.recon_epsilon, x_samples.shape[:-2])
        reconstruction_sigma2 = reconstruction_sigma2.to(reconstruction_mean.device)

        return {
            "x_mu_ps": one_step_decoding_means[...,1:,:],
            "x_Sigma_chol_ps": one_step_decoding_sigma2_chols,
            "y_mu_p": reconstruction_mean,
            "y_Sigma_p": reconstruction_sigma2,
        }

    @staticmethod
    def generate_samples_inner(
        start_samples: _T,
        recurrence_model: RecurrenceModel,
        sigma2xt_schedule: _T,
        recurrence_t_schedule: _T,
        euler_alpha: float,
        device = 'cuda',
        **recurrence_model_kwargs
    ):
        """
        Highly generic, the true schedule is controlled by self.generate_new_samples

        Note that sigmaxt_schedule and recurrence_t_schedule are iterated over together, backwards

        They should be fed in "together and fully" i.e. any clipping that needs to be done should be done upstream

        History however, is fed back in "reverse" time, i.e. most noisy to least noisy!
        """
        (num_samples, hidden_size) = start_samples.shape
        assert hidden_size == recurrence_model.state_space_size
        hidden_samples = start_samples.to(device).unsqueeze(1)
        hidden_sample_history = [hidden_samples.detach()]

        assert (sigma2xt_schedule.shape == recurrence_t_schedule.shape) and (len(recurrence_t_schedule.shape) == 1)

        reversed_sigmaxt_schedule, reversed_recurrence_t_schedule = sigma2xt_schedule.flip(0) ** 0.5, recurrence_t_schedule.flip(0)

        for sigmaxt_tilde, t_tilde in zip(reversed_sigmaxt_schedule, reversed_recurrence_t_schedule):   # t_tilde because time is reversed
            next_means: _T = ((1 - euler_alpha) * hidden_samples) + (euler_alpha * recurrence_model(hidden_samples, t_tilde[None], **recurrence_model_kwargs))
            hidden_samples = next_means + (torch.randn_like(hidden_samples) * sigmaxt_tilde.T)
            hidden_sample_history.append(hidden_samples.detach())
        
        hidden_sample_history = torch.cat(hidden_sample_history, -2).detach()

        return hidden_sample_history, hidden_samples

    @abstractmethod
    def generate_new_samples(
        self,
        base_samples: _T, 
        device = 'cuda', 
        *_, 
        turn_off_noise = False,
        **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def to(self, *args, **kwargs):
        raise NotImplementedError('Need to account for sigma2x and t schedules on a class by class basis')

    # def generate_new_samples(self, base_samples: _T, device = 'cuda', *_, turn_off_noise = False, capped_timesteps = None, **kwargs, ):
    #     (num_samples, hidden_size) = base_samples.shape
    #     hidden_samples = base_samples.to(device).unsqueeze(1)
    #     hidden_sample_history = [hidden_samples.cpu().detach().numpy()]

    #     sigmaxt_schedule = self.sigma2xt_schedule ** 0.5
    #     if capped_timesteps is not None:
    #         assert capped_timesteps <= self.num_hidden_steps
    #     else:
    #         capped_timesteps = self.num_hidden_steps

    #     for t_tilde_idx in range(capped_timesteps - 2, -1, -1):     # In python: T - 2, ..., 0 ---> in maths: T - 1, ..., 1
    #         # [B, dimx] and [B, dimx, dimx] (latter is lower left)
    #         t_tilde = self.recurrence_t_schedule[t_tilde_idx+1, None]
    #         next_mean = ((1 - self.euler_alpha) * hidden_samples) + (self.euler_alpha * self.recurrence_model(hidden_samples, t_tilde, **kwargs))
    #         if turn_off_noise:
    #             hidden_samples = next_mean
    #         else:
    #             next_covar_chol = sigmaxt_schedule[t_tilde_idx]
    #             jitter = torch.randn_like(next_mean)
    #             hidden_samples = next_mean + (jitter * next_covar_chol.T)       # x_{T - t_tilde + 1} so by the end it will be x_{T - (T-2) + 1} = x_0
    #         hidden_sample_history.append(hidden_samples.cpu().detach().numpy())
        
    #     # XXX rehouse this last step!
    #     # t_tilde = self.recurrence_t_schedule[0, None]   # 0
    #     # final_mean = ((1 - self.euler_alpha) * hidden_samples) + (self.euler_alpha * self.recurrence_model(hidden_samples, t_tilde, **kwargs))    # x0
    #     # final_samples = final_mean @ self.behvaiour_projection_matrix # No noise added at final layer! -> y

    #     hidden_sample_history = np.concatenate(hidden_sample_history, -2)

    #     return final_samples.squeeze(1), hidden_sample_history, final_mean.squeeze(1)



class UnrollingDDPMReverseProcess(DDPMReverseProcessBase):

    """
    Designed for single stage diffusion, with unrolling while keeping the time schedule the same at existing steps

    e.g. if my time schedule is currently [dt, 2dt, 3dt, 4dt], I want to unroll one step to [dt, 2dt, 3dt, 4dt]
    """

    sigma2xt_schedule: _T
    num_hidden_steps: int
    ultimate_time_steps: int
    recurrence_t_schedule: _T

    def __init__(
        self,
        hidden_size: int,
        recurrence_model: RecurrenceModel,
        sigma2xt_schedule: _T,
        euler_alpha: float,
        ultimate_time_steps: Optional[int],
        sample_space_size: int = 2,
    ) -> None:
        super().__init__(hidden_size=hidden_size, recurrence_model=recurrence_model, sample_space_size=sample_space_size, euler_alpha=euler_alpha)
        self.sigma2xt_schedule = sigma2xt_schedule.detach().data[:-1]  # Definitely not training on this guy
        self.num_hidden_steps = sigma2xt_schedule.shape[0]
        self.ultimate_time_steps = ultimate_time_steps
        assert ultimate_time_steps >= self.num_hidden_steps
        self.recurrence_t_schedule = torch.linspace(0., 1., self.ultimate_time_steps)[:self.num_hidden_steps]   # p(x_t | x_{t+1}), only needed for t = 1,...,T-1, but not including 1, i.e. T
        assert list(sigma2xt_schedule.shape) == [self.num_hidden_steps]

    def to(self, *args, **kwargs):
        self.sigma2xt_schedule = self.sigma2xt_schedule.to(*args, **kwargs)
        self.recurrence_t_schedule = self.recurrence_t_schedule.to(*args, **kwargs)
        return super(DDPMReverseProcessBase, self).to(*args, **kwargs)
    
    def generate_new_samples(
        self,
        base_samples: _T, 
        device = 'cuda', 
        *_, 
        turn_off_noise = False,
        capped_timesteps = None, 
        **recurrence_model_kwargs
    ):
        """
        This class only has one step of diffusion, and you have the option with "capped_timesteps" to 
        It will end with taking one more step from x_1 to x_0, and projecting to 
        """
        t_schedule = self.recurrence_t_schedule[1:] # i.e. starting at x_2, ..., x_T
        sigma2xt_schedule = self.sigma2xt_schedule

        if turn_off_noise:
            sigma2xt_schedule = sigma2xt_schedule * 0.0

        expected_num_timesteps = self.num_hidden_steps
        if capped_timesteps is not None:
            assert capped_timesteps <= self.num_hidden_steps
            expected_num_timesteps = capped_timesteps + 1
            t_schedule = t_schedule[:capped_timesteps]
            sigma2xt_schedule = sigma2xt_schedule[:capped_timesteps]

        # x_T down to x_1
        hidden_sample_history, x1_sample = self.generate_samples_inner(
            start_samples=base_samples,
            recurrence_model=self.recurrence_model,
            sigma2xt_schedule=sigma2xt_schedule,
            recurrence_t_schedule=t_schedule,
            euler_alpha=self.euler_alpha,
            device = device,
            **recurrence_model_kwargs
        )

        num_samples = base_samples.shape[0]
        assert tuple(hidden_sample_history.shape) == (num_samples, expected_num_timesteps, self.hidden_size), \
            f"Expected hidden_sample_history of shape {(num_samples, expected_num_timesteps, self.hidden_size)}, got {tuple(hidden_sample_history.shape)}"

        # No noise added at this stage
        final_t_tilde = self.recurrence_t_schedule[0]
        x_0_mean = ((1 - self.euler_alpha) * x1_sample) + (self.euler_alpha * self.recurrence_model(x1_sample, final_t_tilde[None], **recurrence_model_kwargs))

        behaviour_samples = x_0_mean @ self.behvaiour_projection_matrix # No noise added at final layer! -> y

        return behaviour_samples.squeeze(1), hidden_sample_history, x_0_mean.squeeze(1)



class MultistageDDPMReverseProcess(DDPMReverseProcessBase):

    """
    Now, we have a fixed number of timesteps (i.e. we are not expecting unrolling like in UnrollingDDPMReverseProcess)

    However, we have a multistage process, where one process passes onto the next.
    They are provided in "forward" order, meaning that for us we pass them in for recursion backwards...
    """

    sigma2xt_schedules: List[_T]
    num_hidden_steps: List[int]
    recurrence_t_schedules: List[_T]

    def __init__(
        self,
        hidden_size: int,
        recurrence_model: RecurrenceModel,
        sigma2xt_schedules: _T,
        sample_space_size: int = 2,
        euler_alpha: float = 0.1
    ) -> None:
        super().__init__(hidden_size=hidden_size, recurrence_model=recurrence_model, sample_space_size=sample_space_size, euler_alpha=euler_alpha)
        
        self.sigma2xt_schedules = []
        self.num_hidden_steps = []
        self.recurrence_t_schedules = []

        for sigma2xt_schedule in sigma2xt_schedules:
            new_sigma2xt_schedule = sigma2xt_schedule.detach().data  # Definitely not training on this guy
            num_hidden_steps = sigma2xt_schedule.shape[0]
            recurrence_t_schedule = torch.linspace(0., 1., self.num_hidden_steps)[:-1]   # p(x_t | x_{t+1}), only needed for t = 1,...,T-1, but not including 1, i.e. T
            
            self.sigma2xt_schedules.insert(0, new_sigma2xt_schedule)
            self.num_hidden_steps.insert(0, num_hidden_steps)
            # self.recurrence_t_schedules.insert(0, recurrence_t_schedule)  # XXX need to count down!

        # x_T no included in recurrence!
        self.sigma2xt_schedules[-1] = self.sigma2xt_schedules[-1][:-1]
        self.num_hidden_steps[-1] = self.num_hidden_steps[-1] - 1
