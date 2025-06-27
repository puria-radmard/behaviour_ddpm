import math

import torch
from torch import Tensor as _T

from typing import Dict, Optional

from ddpm.model.main.base.basic import OneShotDDPMReverseProcess


class ReactionTimeDDPM(OneShotDDPMReverseProcess):
    """
    Built for the case where each sample also comes with a number of timesteps which by we want to sample it.
    """

    def __init__(self, noise_scaler, sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, dt: float, device="cuda"):

        super().__init__(sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, device)

        self.dt = dt    # TODO: make this scheduled also!

        assert tuple(sample_shape) == (1, ), "Cannot have ReactionTimeDDPM with vectoral samples for now"

        assert len(sigma2xt_schedule.unique()) == 1, "GammaTimeVaryingDDPM requires constant sigma2xt_schedule for now!"

        self.noise_scaler = noise_scaler

        if noise_scaler == 'nat':
            pass
        elif isinstance(noise_scaler, float):
            self.noise_scaler_schedule = (
                torch.ones_like(self.noise_scaler_schedule)
                * self.noise_scaler_schedule[1] * noise_scaler
            )
        else:
            raise ValueError(noise_scaler)

    def noise(self, x_0: _T, rts: _T) -> Dict[str, _T | int]:
        """
        Noise from different reaction times

        x_0 is the final decision, i.e. just in {-1, +1}
        rts are the associated reaction times, in number of timesteps
            same shape as x_0 unti last dim, which must be 1. TODO: check shape!

        Noising schedules scaled to condition on reaction times

        Target for the transition ernel is based on the q_posterior mean:

            posterior_mean = mu_q = (x_t - \gamma_t \epsilon_t)
            \gamma_t = \beta_t / \sqrt{1-\bar\alpha_t}

        ...but altered by the euler discretisation:

            mu_q <-> x_t + f(x_t, s_t, t) dt

            ==> f(x_t, s_t, t) <-> (mu_q - x_t) / dt
            
        """
        assert (
            list(x_0.shape[-len(self.sample_shape) :]) == self.sample_shape
        ), f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        num_extra_dim = len(x_0.shape) - len(self.sample_shape)
        x_0 = x_0.unsqueeze(num_extra_dim).expand(
            *x_0.shape[:num_extra_dim],
            self.T,
            *self.sample_shape,
            # *[1] * num_extra_dim, self.T, *[1] * len(self.sample_shape)
        )

        # Temporal smoothing with OU process along the time dimension
        rho = 0.9  # correlation parameter (adjust as needed)
        white_noise = torch.randn_like(x_0)
        epsilon = torch.zeros_like(x_0)
        epsilon[:, :, 0:1, :] = white_noise[:, :, 0:1, :]  # Initialize first timestep
        for t in range(1, self.T):
            epsilon[:, :, t:t+1, :] = rho * epsilon[:, :, t-1:t, :] + math.sqrt(1 - rho**2) * white_noise[:, :, t:t+1, :]
        

        # Only training on noiser end of schedule
        rts_as_indices = (rts / self.dt).int().squeeze(-1)
        rts_as_remaining_indices = self.T - rts_as_indices - 1
        
        relevance_mask = torch.zeros_like(x_0)
        range_tensor = torch.arange(5000, device=relevance_mask.device)
        mask = range_tensor[None, None, :] >= rts_as_remaining_indices[:, :, None]
        relevance_mask[:, :, :, 0] = mask.float()

        # Initialize with NaN
        reshaped_a_t_schedule_rescaled = torch.full_like(relevance_mask, torch.nan)
        reshaped_root_b_t_schedule_rescaled = torch.full_like(relevance_mask, torch.nan)

        # Create range tensor for indexing
        range_tensor = torch.arange(5000, device=relevance_mask.device)  # [5000]

        # Create mask for positions >= r for each (b,c)
        valid_mask = range_tensor[None, None, :] >= rts_as_remaining_indices[:, :, None]  # [4, 1024, 5000]

        # Get the scaling factors - the values at position r for each (b,c)
        scaling_factors_a = self.reshaped_a_t_schedule[rts_as_remaining_indices, 0]  # [4, 1024]

        # Broadcast and rescale
        # self.reshaped_a_t_schedule[None, None, :, :] has shape [1, 1, 5000, 1]
        # scaling_factors_a[:, :, None, None] has shape [4, 1024, 1, 1]
        rescaled_a = self.reshaped_a_t_schedule[None, None, :, :] / scaling_factors_a[:, :, None, None]

        # Apply only where valid (r onwards)
        reshaped_a_t_schedule_rescaled[valid_mask] = rescaled_a[valid_mask]

        # Get the scaling factors for b_t - values at contact point and end
        contact_values_b = self.reshaped_root_b_t_schedule[rts_as_remaining_indices, 0]  # [batch, trials]
        end_value_b = self.reshaped_root_b_t_schedule[self.T-1, 0]  # scalar

        # Affine transform: scale and shift so contact point becomes 0, end stays same
        # new_value = (end_value / (end_value - contact_value)) * (old_value - contact_value)
        scale_factor_b = end_value_b / (end_value_b - contact_values_b[:, :, None, None])  # [batch, trials, 1, 1]
        rescaled_b = scale_factor_b * (self.reshaped_root_b_t_schedule[None, None, :, :] - contact_values_b[:, :, None, None])

        reshaped_root_b_t_schedule_rescaled[valid_mask] = rescaled_b[valid_mask]

        x_t_means = reshaped_a_t_schedule_rescaled * x_0
        x_t = x_t_means + reshaped_root_b_t_schedule_rescaled * epsilon

        # Calculate targets for the DDM dynamics
        # TODO: scale reshaped_base_samples_scaler_schedule and reshaped_residual_scaler_schedule!!!!
        scaled_base_samples = self.reshaped_base_samples_scaler_schedule[*[None]*num_extra_dim] * x_t
        scaled_residual = self.reshaped_residual_scaler_schedule[*[None]*num_extra_dim] * epsilon
        one_step_denoise_mean = scaled_base_samples - scaled_residual       # mu_q      [..., T, <shape samples>]

        kernel_target = (one_step_denoise_mean - x_t) / self.dt

        return {
            "x_t": x_t,
            "epsilon": epsilon,
            "relevance_mask": relevance_mask,
            "rts_as_remaining_indices": rts_as_remaining_indices,
            "kernel_target": kernel_target
        }

    def denoise_one_step(self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float):
        """
        predicted_residual (misnomer) now just acts as the drift term in discrete time DDM, i.e. f(...) in:

        x_t = x_{t-1} + f(x_t, s_t, t) dt + sqrt(dt) v_t \eta
        """
        noise = noise_scaler * torch.randn_like(x_t_plus_1)
        scaled_noise = noise * self.noise_scaler_schedule[-t_idx]

        drift_term = predicted_residual * self.dt
        diffusion_term = scaled_noise * math.sqrt(self.dt)

        x_t = x_t_plus_1 + drift_term + diffusion_term
            
        fake_early_x0_pred = torch.ones_like(x_t) * torch.nan

        return x_t, fake_early_x0_pred
    
