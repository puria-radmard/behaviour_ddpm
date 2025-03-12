import torch
from torch import nn
from torch import Tensor as _T

from typing import Tuple, Optional

from dynamic_observer.model.noise_schedule import ContinuousTimeNoiseSchedule
from dynamic_observer.model.score_approximator import ScoreApproximator
from dynamic_observer.model.discretiser import DynamicsDiscretiser



class ContinuousTimeScoreMatchingDiffusionModel(nn.Module):
    """
    This is another algorithm entirely to DDPMs - these are defined entirely as continuous time SDEs

    Always assuming SDEs of the form
    e.g. variance preserving (VP) SDE - see ContinuousTimeNoiseSchedule docstring
    """
    def __init__(self, sample_dim: int, noise_schedule: ContinuousTimeNoiseSchedule, score_approximator: ScoreApproximator, discretiser: DynamicsDiscretiser) -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.noise_schedule = noise_schedule
        self.score_approximator = score_approximator
        self.discretiser = discretiser

    def run_reverse_dynamics_inner(
        self, start_samples: _T, stimulus: Tuple[_T], time: _T, 
        observations: Optional[_T] = None, projection_matrix: Optional[_T] = None, observation_noise_covar: Optional[_T] = None
    ) -> _T:
        """
        start_samples of shape [..., D]
        each item of stimulus of shape [num_steps, ... (own)] --> reverse time!
        time of shape [num_steps, ...] --> decreasing!

        observation information, if given, also have to have shape [num steps, ... (own)]
            This is because we might consider the case where we have a new end observation at each timestep, e.g. in noisy cue swap model!

        TODO: docstring
        """
        assert start_samples.shape[-1] == self.sample_dim
        assert start_samples.shape[:-1] == time.shape[1:]

        delta_t = - torch.diff(time, dim = 0) # reverse time! [...1, num_steps - 1, 1]
        assert (delta_t > 0.0).all(), "run_reverse_dynamics_inner must be provided with decreasing time - consult docstring for shapes!"
        num_extra_steps = delta_t.shape[0]

        trajectory = [start_samples]

        beta = self.noise_schedule.beta(time)
        # int_t_beta = self.noise_schedule.int_t_beta(time)

        conditioned = (observations is not None)
        if conditioned:
            observation_noise_covar_inverse = torch.linalg.inv(observation_noise_covar)
        
        # In some cases, e.g. the exact case, it's quicker to pass all the stimuli
        # to the score 'approximator' than to pass them just-in-time
        # In other cases, this will just return self

        score_approximator_dispatcher = self.score_approximator.prepare_dispatcher(
            stimuli = stimulus, t = time[:-1]        # Not sure why but [1:] doesn't work here...!
        )

        for t_tilde_idx in range(num_extra_steps):

            t_tilde = time[t_tilde_idx]   # Not sure why but t_tilde_idx + 1 doesn't work here...!
            beta_k = beta[t_tilde_idx].unsqueeze(-1)    # Not sure why but t_tilde_idx + 1 doesn't work here...!
            dt = delta_t[t_tilde_idx].unsqueeze(-1)
            x_k = trajectory[-1]
            step_stimuli = tuple(stim[t_tilde_idx] for stim in stimulus)
            # int_t_beta_k = int_t_beta[t_tilde_idx]

            # [..., D]
            score_approx = score_approximator_dispatcher.approximate_score(x_t = x_k, stimuli = step_stimuli, t = t_tilde, t_tilde_idx = t_tilde_idx)

            if conditioned:
                obs = observations[t_tilde_idx]
                A = projection_matrix[t_tilde_idx]
                obs_covar_inv = observation_noise_covar_inverse[t_tilde_idx]

                conditioned_residual = obs - torch.einsum('...ij,...j->...i', A, x_k.squeeze(-2))
                conditioned_score = torch.einsum('...ij,...ik,...k->...j', A, obs_covar_inv, conditioned_residual)
                score_approx = score_approx + conditioned_score

            drift = - (0.5 * beta_k * x_k) - (beta_k * score_approx)    # f(x, t) - g(t)^2 * s(x, t)
            diffusion = beta_k.sqrt()            # g(t) dW_t
            
            # Replaced with eq 12 in https://arxiv.org/pdf/2011.13456
            # drift = - (0.5 * beta_k * x_k) - ((beta_k * (1.0 - (-2 * int_t_beta_k).exp())) * score_approx)    # f(x, t) - g(t)^2 * s(x, t)
            # diffusion = (beta_k * (1.0 - (-2 * int_t_beta_k).exp())).sqrt()             # g(t) dW_t
            next_step = self.discretiser.step(x_k, - drift, diffusion, dt)    # Reverse time!

            trajectory.append(next_step)

        return torch.stack(trajectory, -2)

    def run_forward_dynamics_inner(
        self, start_samples: _T, time: _T
    ) -> _T:
        """
        start_samples of shape [..., 1, D]
        time of shape [...1, num_steps, 1] --> decreasing!

        TODO: docstring
        """
        assert start_samples.shape[-1] == self.sample_dim
        assert start_samples.shape[:-1] == time.shape[1:]

        delta_t = torch.diff(time, dim = 0) # forward time! [...1, num_steps - 1, 1]
        assert (delta_t > 0.0).all(), "run_forward_dynamics_inner must be provided with increasing time - consult docstring for shapes!"
        num_extra_steps = delta_t.shape[0]

        trajectory = [start_samples]

        beta = self.noise_schedule.beta(time)
        # int_t_beta = self.noise_schedule.int_t_beta(time)

        for t_idx in range(num_extra_steps):

            t = time[t_idx]         # XXX: This should be checked!
            beta_k = beta[t_idx].unsqueeze(-1)
            dt = delta_t[t_idx].unsqueeze(-1)
            x_k = trajectory[-1]
            # int_t_beta_k = int_t_beta[t_idx]

            drift = - (0.5 * beta_k * x_k)                          # f(x, t)
            diffusion = beta_k.sqrt()       # g(t) * dWt
            
            # Replaced with eq 12 in https://arxiv.org/pdf/2011.13456
            # diffusion = (beta_k * (1.0 - (-2 * int_t_beta_k).exp())).sqrt()        # g(t) * dWt

            next_step = self.discretiser.step(x_k, drift, diffusion, dt)
            trajectory.append(next_step)

        return torch.stack(trajectory, -2)

    def run_unconditioned_reverse_dynamics(
        self, start_samples: _T, stimulus: Tuple[_T], num_steps: int, 
        start_time: float = None, end_time: float = 0.0,
    ) -> _T:
        """
        XXX: TODO: ADD STEERING SCORE --> other methods
        aka generate samples

        Use Euler method to run reverse dynamics, evaluating at regular intervals
        """
        if start_time is None:
            start_time = self.noise_schedule.duration

        samples_shape = start_samples.shape         # [..., D]
        assert samples_shape[-1] == self.sample_dim

        assert all(stim.shape[0] == num_steps - 1 for stim in stimulus)

        time = torch.linspace(start_time, end_time, num_steps + 1)[1:]  # [num_steps]
        time = time[:,*[None]*(len(samples_shape)-1)].repeat(1, *start_samples.shape[:-1]) # [num_steps] -> [num_steps, ...1] -> [num_steps, ...]

        return self.run_reverse_dynamics_inner(start_samples, stimulus, time)

    def run_forward_dynamics(
        self, start_samples: _T, num_steps: int, 
        start_time: float = 0.0, end_time: float = None
    ) -> _T:
        """
        XXX: TODO: ADD STIMULUS AND STEERING SCORE --> other methods
        aka devolve to base smoothly

        Use Euler method to run forward dynamics, evaluating at regular intervals
        """
        if end_time is None:
            end_time = self.noise_schedule.duration

        samples_shape = start_samples.shape         # [..., D]
        assert samples_shape[-1] == self.sample_dim

        time = torch.linspace(start_time, end_time, num_steps)  # [num_steps]
        time = time[:,*[None]*(len(samples_shape)-1)].repeat(1, *start_samples.shape[:-1]) # [num_steps] -> [num_steps, ...1] -> [num_steps, ...]

        return self.run_forward_dynamics_inner(start_samples, time)

    def run_conditioned_reverse_dynamics(
        self, start_samples: _T, stimulus: Tuple[_T], num_steps: int, 
        observations: _T, projection_matrix: _T, observation_noise_covar: _T,
        start_time: float = None, end_time: float = 0.0,
    ) -> _T:
        """
        aka generate samples

        Use Euler method to run reverse dynamics, evaluating at regular intervals
        """
        if start_time is None:
            start_time = self.noise_schedule.duration

        samples_shape = start_samples.shape         # [..., D]
        assert samples_shape[-1] == self.sample_dim

        assert all(stim.shape[0] == num_steps - 1 for stim in stimulus)
        assert observations.shape[0] == projection_matrix.shape[0] == observation_noise_covar.shape[0] == num_steps - 1

        time = torch.linspace(start_time, end_time, num_steps + 1)[1:]  # [num_steps]  # [num_steps]
        time = time[:,*[None]*(len(samples_shape)-1)].repeat(1, *start_samples.shape[:-1]) # [num_steps] -> [num_steps, ...1] -> [num_steps, ...]

        return self.run_reverse_dynamics_inner(
            start_samples, stimulus, time,
            observations = observations, projection_matrix = projection_matrix, observation_noise_covar = observation_noise_covar,
        )

