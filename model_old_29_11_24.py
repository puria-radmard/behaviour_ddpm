from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T
import torch.nn.functional as F

from typing import List, Dict, Optional


class TimeSensitiveNoiseProcessCovarGenerator(nn.Module):
    """
    Takes you from some time to the Cholesky of the (reverse-time) dynamics noise
    """
    def __init__(self, output_sizes: List[int], state_space_size: int):
        
        super().__init__()

        self.output_sizes = output_sizes.copy()
        self.state_space_size = state_space_size

        if len(output_sizes):
            phi_model_layers = [nn.Linear(2, output_sizes[0]), nn.ReLU()]    # takes in sigma2_t and t/T
            for i, op_s in enumerate(output_sizes[1:]):
                phi_model_layers.extend([nn.Linear(output_sizes[i], op_s), nn.ReLU()])
            phi_model_layers = phi_model_layers[:-1]
            self.phi_model = nn.Sequential(*phi_model_layers)
        else:
            self.phi_model = nn.Identity()

        self.final_output_size = int(0.5 * state_space_size * (state_space_size + 1))
        self.output_layer = nn.Linear(output_sizes[-1], self.final_output_size)

    def forward_phi(self, sigma2_schedule: _T, t_schedule: _T) -> _T:
        assert sigma2_schedule.shape == t_schedule.shape, (sigma2_schedule.shape, t_schedule.shape)
        assert (0<=t_schedule).all() and (t_schedule<1).all(), (t_schedule.min(), t_schedule.max())
        inp = torch.stack([sigma2_schedule, t_schedule], -1)
        return self.phi_model(inp)
    
    def forward(self, sigma2_schedule: _T, t_schedule: _T) -> _T:
        """
        Both inputs of size [...,T-1] (T = total time)
        """
        phi_output = self.forward_phi(sigma2_schedule, t_schedule)
        flat_cholesky = self.output_layer(phi_output)
        output = torch.zeros(*flat_cholesky.shape[:-1], self.state_space_size, self.state_space_size, device = flat_cholesky.device)
        tril_indices = torch.tril_indices(row=self.state_space_size, col=self.state_space_size, offset=0)
        output[..., *tril_indices] = flat_cholesky
        output[..., range(self.state_space_size), range(self.state_space_size)] = F.softplus(output[..., range(self.state_space_size), range(self.state_space_size)])
        return output

    def impart_all_but_last_weights(self, other: TimeSensitiveNoiseProcessCovarGenerator) -> None:
        self.phi_model.load_state_dict(other.phi_model.state_dict())

    def do_dng(self, n_idx: int):
        raise NotImplementedError('Need to account for softplus!')

        with torch.no_grad():
            assert 0<=n_idx<self.state_space_size

            current_last_weight = self.output_layer.weight     # [d(d+1) / 2, h]
            current_last_bias = self.output_layer.bias         # [d(d+1) / 2]

            tril_indices_previous = torch.tril_indices(row=self.state_space_size, col=self.state_space_size, offset=0)
            tril_indices_new = torch.tril_indices(row=self.state_space_size+1, col=self.state_space_size+1, offset=0)
            duplicated_indices = tril_indices_previous[:,n_idx]

            new_weight_reshaped = torch.zeros(self.state_space_size + 1, self.state_space_size + 1, current_last_weight.shape[-1], device = current_last_weight.device)
            new_weight_reshaped[*tril_indices_previous,:] = current_last_weight
            new_weight_reshaped[-1,-1,:] = new_weight_reshaped[*duplicated_indices,:]
            new_weight = new_weight_reshaped[*tril_indices_new, :]

            new_bias_reshaped = torch.zeros(self.state_space_size + 1, self.state_space_size + 1, device = current_last_bias.device)
            new_bias_reshaped[*tril_indices_previous] = current_last_bias
            new_bias_reshaped[-1,-1] = new_bias_reshaped[*duplicated_indices]
            new_bias = new_bias_reshaped[*tril_indices_new]

            new_model = TimeSensitiveNoiseProcessCovarGenerator(self.output_sizes, self.state_space_size + 1)
            self.impart_all_but_last_weights(new_model)
            new_model.output_layer.weight.data = new_weight
            new_model.output_layer.bias.data = new_bias


            # Test cases
            test_batch_size = 100
            own_phi_output = self.forward_phi(torch.linspace(0, 1, test_batch_size), torch.linspace(0,1,test_batch_size+1)[:-1])
            new_phi_output = new_model.forward_phi(torch.linspace(0, 1, test_batch_size), torch.linspace(0,1,test_batch_size+1)[:-1])
            assert torch.isclose(own_phi_output, new_phi_output).all() 

            test_vector = own_phi_output # torch.randn(1, 512)
            own_output = test_vector @ current_last_weight.T + current_last_bias
            new_output = test_vector @ new_weight.T + new_bias
            reshaped_own_output = torch.zeros(test_batch_size, self.state_space_size, self.state_space_size)
            reshaped_own_output[:,*tril_indices_previous] = own_output
            reshaped_new_output = torch.zeros(test_batch_size, 1 + self.state_space_size, 1 + self.state_space_size)
            reshaped_new_output[:,*tril_indices_new] = new_output
            assert torch.isclose(reshaped_own_output, reshaped_new_output[:,:-1,:-1]).all()

            own_full_output = self.forward(torch.linspace(0, 1, test_batch_size), torch.linspace(0,1,test_batch_size+1)[:-1])
            new_full_output = new_model.forward(torch.linspace(0, 1, test_batch_size), torch.linspace(0,1,test_batch_size+1)[:-1])
            assert torch.isclose(own_full_output, new_full_output[:,:-1,:-1]).all()

        return new_model



class TimeConstantNoiseProcessCovarGenerator(nn.Module):

    flat_noise_sigma_chol: _T

    def __init__(self, state_space_size: int):
        super().__init__()
        self.state_space_size = state_space_size
        self.final_output_size = int(0.5 * state_space_size * (state_space_size + 1))
        self.register_parameter('flat_noise_sigma_chol', nn.Parameter(torch.randn(self.final_output_size).float() / self.state_space_size, requires_grad = True))

    def forward(self, sigma2_schedule: _T, t_schedule: _T) -> _T:
        assert sigma2_schedule.shape == t_schedule.shape, (sigma2_schedule.shape, t_schedule.shape)
        if len(sigma2_schedule) >= 1:
            assert (sigma2_schedule == sigma2_schedule[0]).all(), "TimeConstantNoiseProcessCovarGenerator should not be used for non-constant forward time noise injection!"
        output = torch.zeros(*sigma2_schedule.shape, self.state_space_size, self.state_space_size, device = sigma2_schedule.device)
        tril_indices = torch.tril_indices(row=self.state_space_size, col=self.state_space_size, offset=0)
        output[..., *tril_indices] = self.flat_noise_sigma_chol
        output[..., range(self.state_space_size), range(self.state_space_size)] = F.softplus(output[..., range(self.state_space_size), range(self.state_space_size)])
        return output

    def impart_all_but_last_weights(self, other: TimeConstantNoiseProcessCovarGenerator) -> None:
        with torch.no_grad():
            assert self.flat_noise_sigma_chol.data.shape == other.flat_noise_sigma_chol.data.shape
            self.flat_noise_sigma_chol.data = other.flat_noise_sigma_chol.data
    
    def do_dng(self, n_idx: int):
        raise NotImplementedError



class RecurrenceModel(nn.Module):

    def __init__(self, state_space_size: int, recurrence_hidden_layers: List[int]) -> None:
        super().__init__()
        self.state_space_size = state_space_size
        self.recurrence_hidden_layers = recurrence_hidden_layers
        all_layer_sizes = recurrence_hidden_layers + [state_space_size]
        recurrence_layers = [nn.Linear(state_space_size + 1, all_layer_sizes[0]), nn.Softplus()]        # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            recurrence_layers.extend([nn.Linear(all_layer_sizes[i], op_s), nn.Softplus()])
        recurrence_layers = recurrence_layers[:-1] + [nn.ReLU()]  # Final output is rates
        self.model = nn.Sequential(*recurrence_layers)   # R^N -> R^N_+

    def forward(self, x: _T, t_schedule: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_schedule of shape [T]
        """
        assert (x.shape[-2],) == tuple(t_schedule.shape)
        reshaped_t_schedule = t_schedule.reshape(*[1 for _ in range(len(x.shape) - 2)], -1, 1).repeat(*x.shape[:-2], 1, 1)
        x_concat = torch.concat([x, reshaped_t_schedule], -1)
        return self.model(x_concat)

    def do_dng(self, n_idx):
        raise NotImplementedError


class DDPMReverseProcess(nn.Module):
    """
    No sensory input at the moment!
    """
    
    recon_epsilon: float = 0.01
    
    sigma2xt_schedule: _T

    def __init__(
        self,
        hidden_size: int,
        sigma2xt_schedule: _T,
        ultimate_time_steps: int,
        noise_model: TimeSensitiveNoiseProcessCovarGenerator | TimeConstantNoiseProcessCovarGenerator,
        recurrence_model: RecurrenceModel,
        sample_space_size: int = 2,
        euler_alpha: float = 0.1,
        learnable_linking_matrix: bool = False
    ) -> None:
        
        super().__init__()

        self.learnable_linking_matrix = learnable_linking_matrix
        if learnable_linking_matrix:
            raise Exception
            self.register_parameter('linking_matrix', nn.Parameter(torch.randn(sample_space_size, hidden_size) / (hidden_size**0.5)))
        elif hidden_size == sample_space_size:
            self.register_buffer('linking_matrix', torch.eye(hidden_size), persistent=True)   # [dim y, dim x]
        elif hidden_size > sample_space_size:
            gaus = torch.randn(hidden_size, hidden_size)
            svd = torch.linalg.svd(gaus)
            orth = svd[0] @ svd[2]
            self.register_buffer('linking_matrix', orth[:sample_space_size], persistent=True)   # [dim y, dim x]
        else:
            raise NotImplementedError

        assert recurrence_model.state_space_size == hidden_size == noise_model.state_space_size
        self.noise_model = noise_model
        self.recurrence_model = recurrence_model
        
        self.euler_alpha = euler_alpha
        self.sample_space_size = sample_space_size
        self.hidden_size = hidden_size
        
        self.full_sigma2xt_schedule = sigma2xt_schedule.detach().data
        self.sigma2xt_schedule = sigma2xt_schedule.detach().data[:-1]  # Definitely not training on this guy
        self.num_hidden_steps = sigma2xt_schedule.shape[0]
        self.ultimate_time_steps = ultimate_time_steps
        assert ultimate_time_steps >= self.num_hidden_steps

        self.noise_t_schedule = torch.linspace(0., 1., self.num_hidden_steps)[:-1]
        self.recurrence_t_schedule = torch.linspace(0., 1., self.ultimate_time_steps)[:self.num_hidden_steps]   # p(x_t | x_{t+1}), only needed for t = 1,...,T-1, but not including 1, i.e. T

        # self.final_sigma2 = sigma2xt_schedule[-1]     # XXX calculated final marginal!
        assert list(sigma2xt_schedule.shape) == [self.num_hidden_steps]

    def to(self, *args, **kwargs):
        self.sigma2xt_schedule = self.sigma2xt_schedule.to(*args, **kwargs)
        self.noise_t_schedule = self.noise_t_schedule.to(*args, **kwargs)
        self.recurrence_t_schedule = self.recurrence_t_schedule.to(*args, **kwargs)
        if not self.learnable_linking_matrix:
            self.linking_matrix = self.linking_matrix.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def do_dng(self, n_idx: int):

        assert 0<=n_idx<self.hidden_size
        
        new_noise_model = self.noise_model.do_dng(n_idx)
        new_recurrence_model = self.recurrence_model.do_dng(n_idx)

        with torch.no_grad():

            new_model = DDPMReverseProcess(
                hidden_size = self.hidden_size + 1,
                sigma2xt_schedule = self.full_sigma2xt_schedule,
                noise_model = new_noise_model,
                recurrence_model = new_recurrence_model,
                sample_space_size = self.sample_space_size,
                euler_alpha = self.euler_alpha
            )

            # # # New neuron pushed to final element
            # # new_recurrent_linear_weight = torch.cat([
            # #     self.recurrent_linear.weight.data[:n_idx],
            # #     self.recurrent_linear.weight.data[n_idx].unsqueeze(0),
            # #     self.recurrent_linear.weight.data[n_idx+1:],
            # #     self.recurrent_linear.weight.data[n_idx].unsqueeze(0),
            # # ], dim = 0)
            # # new_recurrent_linear_weight = torch.cat([
            # #     new_recurrent_linear_weight[:,:n_idx],
            # #     0.5 * new_recurrent_linear_weight[:,n_idx].unsqueeze(1),
            # #     new_recurrent_linear_weight[:,n_idx+1:],
            # #     0.5 * new_recurrent_linear_weight[:,n_idx].unsqueeze(1),
            # # ], dim = 1)

            # # new_recurrent_linear_bias = torch.cat([
            # #     self.recurrent_linear.bias.data[:n_idx],
            # #     0.5 * self.recurrent_linear.bias.data[n_idx].unsqueeze(0),
            # #     self.recurrent_linear.bias.data[n_idx+1:],
            # #     0.5 * self.recurrent_linear.bias.data[n_idx].unsqueeze(0),
            # # ])

            # # new_linking_matrix = torch.cat([
            # #     self.linking_matrix.data.data[:,:n_idx],
            # #     self.linking_matrix.data.data[:,n_idx].unsqueeze(1),
            # #     self.linking_matrix.data.data[:,n_idx+1:],
            # #     self.linking_matrix.data.data[:,n_idx].unsqueeze(1),
            # # ], dim = -1)

            # # new_model.recurrent_linear.weight.data = new_recurrent_linear_weight
            # # new_model.recurrent_linear.bias.data = new_recurrent_linear_bias
            # # new_model.linking_matrix.data = new_linking_matrix

        return new_model

    @staticmethod
    def unsqueeze_for_arbitrary_front_dims(tensor: _T, extra_dims: List[int]) -> _T:
        return tensor[*[None]*len(extra_dims)].repeat(*extra_dims, *[1]*len(tensor.shape))

    @property
    def noise_covar_chol(self) -> _T:
        """[T-1, dimx, dimx]"""
        sigma2_schedule = self.sigma2xt_schedule  # [T-1]
        t_schedule = self.noise_t_schedule    # [T-1]
        noise_covar_chol = self.noise_model(sigma2_schedule = sigma2_schedule, t_schedule = t_schedule)
        assert list(noise_covar_chol.shape) == [self.num_hidden_steps - 1, self.hidden_size, self.hidden_size], noise_covar_chol.shape
        return noise_covar_chol
    
    @property
    def noise_covar(self) -> _T:
        raise TypeError('Use DDPMReverseProcess.noise_covar_chol always!')
        """[T-1, dimx, dimx]"""
        noise_covar_chol = self.noise_covar_chol
        return torch.bmm(noise_covar_chol, noise_covar_chol.transpose(1, 2))

    @property
    def behvaiour_projection_matrix(self,) -> _T:
        # if self.tie_behvaiour_projection_matrix:
        #     try:
        #         linking_matrix_penrose = torch.inverse(self.linking_matrix.T @ self.linking_matrix) @ self.linking_matrix.T
        #     except:
        #         inverse_offset = torch.eye(self.linking_matrix.shape[1], device = self.linking_matrix.device, dtype = self.linking_matrix.dtype) * 1e-6
        #         linking_matrix_penrose = torch.inverse(self.linking_matrix.T @ self.linking_matrix + inverse_offset) @ self.linking_matrix.T
        #     return linking_matrix_penrose
        # else:
        #     return self.behvaiour_projection_matrix_raw
        return torch.linalg.inv(self.linking_matrix.T @ self.linking_matrix) @ self.linking_matrix.T   # ([dim x, dim y] @ [dim y, dim x]) ^2 @ [dim x, dim y] -> [dim x, dim y]

    def one_step_reverse_generative_process(self, x_samples: _T) -> Dict[str, _T]:
        """
        One shot forward disitribution prediction

        x_samples is output of run_forward_process, shaped [..., T, dim_x]
            x_samples[..., 1:, :] used to generate one step predictions, which are kept in that order
            x_samples[..., 0, :] used to generate decoding predictions

        outputs:
            x_mu_ps [..., T-1, nh]
            x_Sigma_chol_ps [..., T-1, nh, nh], lower-left
            y_mu_p [..., dim_y (2)]
            y_Sigma_p [..., dim_y, dim_y]
        
        TODO: MAKE y_Sigma_p [...] SINCE IT'S JUST DIAGONAL - ALSO FIX THE CROSS ENTROPY FUNCTION
        """
        assert list(x_samples.shape[-2:]) == [self.num_hidden_steps, self.hidden_size], f"x_samples of shape {x_samples.shape} don't match {[self.num_hidden_steps, self.hidden_size]} in last two dimensions"

        # [..., T, dim_x] (to be reduced) and [..., T-1, dim_x, dim_x]
        one_step_decoding_means = ((1 - self.euler_alpha) * x_samples) + (self.euler_alpha * self.recurrence_model(x_samples, self.recurrence_t_schedule))
        #one_step_decoding_sigma2s = self.noise_covar(x_samples.shape[:-1])   # Same everywhere, even across timesteps!
        #one_step_decoding_sigma2s = one_step_decoding_sigma2s[...,1:,:,:]
        one_step_decoding_sigma2s = self.unsqueeze_for_arbitrary_front_dims(self.noise_covar_chol, x_samples.shape[:-2])

        # [..., dim_y = 2] and [..., dim_y, dim_y]
        reconstruction_mean = one_step_decoding_means[...,0,:] @ self.behvaiour_projection_matrix
        reconstruction_sigma2 = self.unsqueeze_for_arbitrary_front_dims(torch.eye(self.sample_space_size) * self.recon_epsilon, x_samples.shape[:-2])
        reconstruction_sigma2 = reconstruction_sigma2.to(reconstruction_mean.device)

        return {
            "x_mu_ps": one_step_decoding_means[...,1:,:],
            "x_Sigma_chol_ps": one_step_decoding_sigma2s,
            "y_mu_p": reconstruction_mean,
            "y_Sigma_p": reconstruction_sigma2,
        }

    def generate_new_samples(self, num_samples: int, base_samples: Optional[_T], device = 'cuda'):
        if base_samples is None:
            raise TypeError('Havent done base measure calculation yet!')
            hidden_samples = torch.randn(num_samples, self.hidden_size).to(device) * self.final_sigma2 ** 0.5    # x_T
        else:
            assert tuple(base_samples.shape) == (num_samples, self.hidden_size)
            hidden_samples = base_samples.to(device).unsqueeze(1)
        noise_covar_chol = self.noise_covar_chol  # [T-1, dimx, dimx]
        for t_tilde_idx in range(len(self.recurrence_t_schedule) - 2, -1, -1):     # In python: T - 2, ..., 0 ---> in maths: T - 1, ..., 1
            # [B, dimx] and [B, dimx, dimx] (latter is lower left)
            t_tilde = self.recurrence_t_schedule[t_tilde_idx+1, None]
            next_mean = ((1 - self.euler_alpha) * hidden_samples) + (self.euler_alpha * self.recurrence_model(hidden_samples, t_tilde))
            next_covar_chol = noise_covar_chol[t_tilde_idx]
            jitter = torch.randn_like(next_mean)
            hidden_samples = next_mean + (jitter @ next_covar_chol.T)       # x_{T - t_tilde + 1} so by the end it will be x_{T - (T-2) + 1} = x_0
        t_tilde = self.recurrence_t_schedule[0, None]   # 0
        final_mean = ((1 - self.euler_alpha) * hidden_samples) + (self.euler_alpha * self.recurrence_model(hidden_samples, t_tilde))    # x0
        final_samples = final_mean @ self.behvaiour_projection_matrix # No noise added at final layer! -> y
        return final_samples.squeeze(1)
