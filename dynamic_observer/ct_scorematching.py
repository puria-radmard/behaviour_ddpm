from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from tqdm import tqdm

from typing import Dict, Tuple, Optional


from abc import ABC, abstractmethod



class ContinuousTimeNoiseSchedule(ABC):
    """
    Noise schedule beta(t)

    Assume variance preserving (VP) SDE:
        dx = - 0.5 beta(t) x dt + sqrt(beta(t)) dWt
    
    throughout, so that:
        x_t | x_0 ~ N(mu_t * x_0, sigma_t^2 I)

    where:
        mu_t = exp(-0.5 * \int_0^t beta(s) ds)
        sigma_t^2 = 1 - exp(- \int_0^t beta(s) ds) = 1 - mu_t^2

    Therefore, this class needs beta(t) and int_t_beta = \int_0^t beta(s)
        to be defined, so that mu_t and sigma_t can be returned with boilerplate
    Also apply noise

    Furthermore, if we know that:
        x_0 ~ N(m, S)
    we can find p(x_t) = <p(x_t | x_0)>_{p(x_0)}
    
    XXX: This was first done in ddpm.tasks.distribution.DistributionInformation,
        but repeated here for cleaner code :/
    """

    duration: float

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def beta(self, time: _T) -> _T:
        """
        Returns same shape as time
        """
        raise NotImplementedError
    
    @abstractmethod
    def int_t_beta(self, time: _T) -> _T:
        """
        Returns same shape as time
        """
        raise NotImplementedError

    @torch.no_grad()
    def noising_factor(self, time: _T) -> _T:
        """
        exp(-0.5 * \int_0^t beta(s) ds), see class docstring
        """
        int_t_beta = self.int_t_beta(time)
        exp = (-0.5 * int_t_beta).exp()
        return exp
    
    @torch.no_grad()
    def noise_and_conditional_score(self, x0: _T, time: _T) -> Dict[str, _T]:
        """
        time and x0 have to have broadcastable shapes
        """
        noising_factor = self.noising_factor(time = time)
        epsilon = torch.randn_like(x0)
        xt = (noising_factor * x0) + ((1.0 - noising_factor.square()).sqrt() * epsilon)
        conditional_score = - epsilon / (1.0 - noising_factor.square()).sqrt()                 # XXX: Check before training!
        return {
            'noising_factor': noising_factor,
            'epsilon': epsilon,
            'xt': xt,
            'conditional_score': conditional_score
        }

    @torch.no_grad()
    def marginal_moments_gaussian_gt_distribution(self, m_x0: _T, S_x0: _T, time: _T) -> Dict[str, _T]:
        """
        Integrate p(x_0) = N(m_x0, S_x0) out of conditional moments, see class docstring
        m_x0 of shape [..., d]
        S_x0 of shape [..., d, d]
        time has a broadcastable shape with m_x0

        TODO: make this multimodal?
        """
        noising_factor = self.noising_factor(time = time)
        reshaped_noising_factor = noising_factor.unsqueeze(-1)
        iden = torch.eye(S_x0.shape[-1])[*[None]*(len(S_x0.shape) - 2)].to(S_x0.device)
        extra_var = (1.0 - reshaped_noising_factor.square()) * iden
        S_xt = (reshaped_noising_factor.square() * S_x0) + extra_var
        return {
            'm_xt': m_x0 * noising_factor,
            'S_xt': S_xt
        }

    @torch.no_grad()
    def marginal_score(self, m_xt: _T, S_xt: _T, xt: _T) -> _T:
        """
        xt and m_x0 of shape [..., d]
        S_x0 of shape [..., d, d]

        returns [..., d]
        """
        S_xt_inv = torch.linalg.inv(S_xt)
        residual = (xt - m_xt)
        score = - torch.einsum('...ij,...j->...i', S_xt_inv, residual)
        return score

    @torch.no_grad()
    def summarise_noising_factor(self, num_points: int = 100) -> Tuple[_T, _T]:
        time = torch.linspace(0, self.duration, num_points)
        return time, self.noising_factor(time=time)



class ConstantTimeNoiseSchedule(ContinuousTimeNoiseSchedule):

    def __init__(self, noise_level: float, duration: float = 10.0) -> None:
        super().__init__()
        self.noise_level = noise_level
        self.duration = duration
        
        smallest_scaling_factor = self.summarise_noising_factor()[1][-1]
        assert smallest_scaling_factor > 0.001, smallest_scaling_factor.item()

    def beta(self, time: _T) -> _T:
        """
        Returns same shape as time
        """
        main = torch.ones_like(time) * self.noise_level
        return main
    
    def int_t_beta(self, time: _T) -> _T:
        """
        Returns same shape as time
        """
        return time * self.noise_level


class LinearIncreaseNoiseSchedule(ContinuousTimeNoiseSchedule):
    
    def __init__(self, start_noise_level: float, end_noise_level: float, duration: float = 10.0) -> None:
        super().__init__()
        self.start_noise_level = start_noise_level    # at time 0
        self.end_noise_level = end_noise_level  # at time 1
        self.slope = end_noise_level - start_noise_level
        self.duration = duration

        smallest_scaling_factor = self.summarise_noising_factor()[1][-1]
        assert smallest_scaling_factor > 0.001, smallest_scaling_factor.item()
    
    def beta(self, time: _T) -> _T:
        return self.start_noise_level + self.slope * time

    def int_t_beta(self, time: _T) -> _T:
        return (
            self.start_noise_level * time
            + 0.5 * self.slope * time.square()
        )



class ScoreApproximatorDispatcher(ABC):

    @abstractmethod
    def approximate_score(self, x_t: _T, stimuli: Tuple[_T], t: _T, t_idx: _T, **kwargs):
        """
        TODO: document!
        """
        raise NotImplementedError



class TrueScoreApproximatorDispatcher(ScoreApproximatorDispatcher):

    def __init__(self, stimuli: Tuple[_T, ...], t: _T, noise_schedule: ContinuousTimeNoiseSchedule) -> None:
        num_steps = t.shape[0]
        assert all(stim.shape[0] == num_steps for stim in stimuli)
        assert (t.shape[0],) == (num_steps,)

        reshaped_m_x0, reshaped_S_x0 = stimuli
        assert len(reshaped_m_x0.shape) == 3 and len(reshaped_S_x0.shape) == 4
        reshaped_m_x0 = reshaped_m_x0.unsqueeze(-2)
        reshaped_S_x0 = reshaped_S_x0.unsqueeze(-3)

        t = t.reshape(num_steps, *[1]* (len(reshaped_m_x0.shape) - 1))

        self.noise_schedule = noise_schedule
        marginal_moments = self.noise_schedule.marginal_moments_gaussian_gt_distribution(reshaped_m_x0, reshaped_S_x0, t)

        self.m_xt = marginal_moments['m_xt']    # [num steps,... D]
        self.S_xt = marginal_moments['S_xt']    # [num steps,... D, D]

    def approximate_score(self, x_t: _T, t_tilde_idx: int, **kwargs):
        m_xt = self.m_xt[t_tilde_idx]
        S_xt = self.S_xt[t_tilde_idx]
        return self.noise_schedule.marginal_score(m_xt, S_xt, x_t)


class ScoreApproximator(nn.Module, ABC):

    @abstractmethod
    def approximate_score(self, x_t: _T, stimuli: Tuple[_T], t: _T, **kwargs):
        """
        x_t comes in shape [..., D]
        t comes in all ones, with same number of axes
        """
        raise NotImplementedError

    def prepare_dispatcher(self, stimuli: Tuple[_T], t: _T) -> ScoreApproximator | ScoreApproximatorDispatcher:
        return self


class TrueScore(ScoreApproximator):
    """
    By giving this `approximator` access to the (TODO) true embedding distribution and the
    noise schedule, it can find the marginal distribution, and therefore score, of the noised
    data

    TODO: above - for now it just runs to the same fixed point everytime!
    """

    def __init__(self, noise_schedule: ContinuousTimeNoiseSchedule) -> None:
        super().__init__()
        self.noise_schedule = noise_schedule

    def prepare_dispatcher(self, stimuli: Tuple[_T, ...], t: _T) -> ScoreApproximator | ScoreApproximatorDispatcher:
        return TrueScoreApproximatorDispatcher(stimuli, t, self.noise_schedule)

    def approximate_score(self, x_t: _T, stimuli: Tuple[_T, ...], t: _T, **kwargs):
        """
        Should not be directly accessed anymore!
        """
        reshaped_m_x0, reshaped_S_x0 = stimuli
        assert len(reshaped_m_x0.shape) == 2 and len(reshaped_S_x0.shape) == 3
        reshaped_m_x0 = reshaped_m_x0.unsqueeze(1)
        reshaped_S_x0 = reshaped_S_x0.unsqueeze(1)
        marginal_moments = self.noise_schedule.marginal_moments_gaussian_gt_distribution(reshaped_m_x0, reshaped_S_x0, t)
        m_xt = marginal_moments['m_xt']
        S_xt = marginal_moments['S_xt']
        return self.noise_schedule.marginal_score(m_xt, S_xt, x_t)





class DynamicsDiscretiser(ABC):

    @abstractmethod
    def step(self, x_k: _T, drift: _T, diffusion: _T, dt: _T, noise: Optional[_T] = None) -> _T:
        raise NotImplementedError
    

class EulerDiscretiser(DynamicsDiscretiser):

    def step(self, x_k: _T, drift: _T, diffusion: _T, dt: _T, noise: Optional[_T] = None) -> _T:
        if noise is None:
            noise = torch.randn_like(x_k)
        return x_k + (drift * dt) + (diffusion * dt.sqrt() * noise)



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
        start_samples of shape [..., 1, D]
        each item of stimulus of shape [num_steps, ... (own)] --> reverse time!
        time of shape [...1, num_steps, 1] --> decreasing!

        observation information, if given, also have to have shape [num steps, ... (own)]
            This is because we might consider the case where we have a new end observation at each timestep, e.g. in noisy cue swap model!

        TODO: docstring
        """
        assert tuple(start_samples.shape[-2:]) == (1, self.sample_dim)
        assert time.shape[-1] == 1

        delta_t = - torch.diff(time, dim = len(time.shape)-2) # reverse time! [...1, num_steps - 1, 1]
        assert (delta_t > 0.0).all(), "run_reverse_dynamics_inner must be provided with decreasing time - consult docstring for shapes!"
        num_extra_steps = delta_t.shape[-2]

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
            stimuli = stimulus, t = time[0, ..., :-1, 0]        # Not sure why but [1:] doesn't work here...!
        )

        for t_tilde_idx in tqdm(range(num_extra_steps)):

            t_tilde = time[..., [t_tilde_idx], :]   # Not sure why but t_tilde_idx + 1 doesn't work here...!
            beta_k = beta[..., [t_tilde_idx], :]    # Not sure why but t_tilde_idx + 1 doesn't work here...!
            dt = delta_t[..., [t_tilde_idx], :]
            x_k = trajectory[-1]
            step_stimuli = tuple(stim[t_tilde_idx] for stim in stimulus)
            # int_t_beta_k = int_t_beta[..., [t_tilde_idx], :]

            score_approx = score_approximator_dispatcher.approximate_score(x_t = x_k, stimuli = step_stimuli, t = t_tilde, t_tilde_idx = t_tilde_idx)

            if conditioned:
                obs = observations[t_tilde_idx]
                A = projection_matrix[t_tilde_idx]
                obs_covar_inv = observation_noise_covar_inverse[t_tilde_idx]

                assert x_k.shape[-2] == 1

                conditioned_residual = obs - torch.einsum('...ij,...j->...i', A, x_k.squeeze(-2))
                conditioned_score = torch.einsum('...ij,...ik,...k->...j', A, obs_covar_inv, conditioned_residual)
                score_approx = score_approx + conditioned_score.unsqueeze(-2)

            drift = - (0.5 * beta_k * x_k) - (beta_k * score_approx)    # f(x, t) - g(t)^2 * s(x, t)
            diffusion = beta_k.sqrt()            # g(t) dW_t
            
            # Replaced with eq 12 in https://arxiv.org/pdf/2011.13456
            # drift = - (0.5 * beta_k * x_k) - ((beta_k * (1.0 - (-2 * int_t_beta_k).exp())) * score_approx)    # f(x, t) - g(t)^2 * s(x, t)
            # diffusion = (beta_k * (1.0 - (-2 * int_t_beta_k).exp())).sqrt()             # g(t) dW_t

            next_step = self.discretiser.step(x_k, - drift, diffusion, dt)    # Reverse time!
            trajectory.append(next_step)

        return torch.concat(trajectory, -2)

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
        time = time.unsqueeze(-1)[*[None]*(len(samples_shape)-1)] # [num_steps] -> [num_steps, 1] -> [...1, num_steps, 1]
        start_samples = start_samples.unsqueeze(-2) # [..., 1, D]

        return self.run_reverse_dynamics_inner(start_samples, stimulus, time)

    def run_forward_dynamics_inner(
        self, start_samples: _T, time: _T
    ) -> _T:
        """
        start_samples of shape [..., 1, D]
        time of shape [...1, num_steps, 1] --> decreasing!

        TODO: docstring
        """
        assert tuple(start_samples.shape[-2:]) == (1, self.sample_dim)
        assert time.shape[-1] == 1

        delta_t = torch.diff(time, dim = len(time.shape)-2) # forward time! [...1, num_steps - 1, 1]
        assert (delta_t > 0.0).all(), "run_forward_dynamics_inner must be provided with increasing time - consult docstring for shapes!"
        num_extra_steps = delta_t.shape[-2]

        trajectory = [start_samples]

        beta = self.noise_schedule.beta(time)
        # int_t_beta = self.noise_schedule.int_t_beta(time)

        for t_idx in range(num_extra_steps):

            t = time[..., [t_idx + 1], :]
            beta_k = beta[..., [t_idx + 1], :]
            dt = delta_t[..., [t_idx], :]
            x_k = trajectory[-1]
            # int_t_beta_k = int_t_beta[..., [t_idx], :]

            drift = - (0.5 * beta_k * x_k)                          # f(x, t)
            diffusion = beta_k.sqrt()       # g(t) * dWt
            
            # Replaced with eq 12 in https://arxiv.org/pdf/2011.13456
            # diffusion = (beta_k * (1.0 - (-2 * int_t_beta_k).exp())).sqrt()        # g(t) * dWt

            next_step = self.discretiser.step(x_k, drift, diffusion, dt)
            trajectory.append(next_step)

        return torch.concat(trajectory, -2)

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
        time = time.unsqueeze(-1)[*[None]*(len(samples_shape)-1)] # [num_steps] -> [num_steps, 1] -> [...1, num_steps, 1]
        start_samples = start_samples.unsqueeze(-2) # [..., 1, D]

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
        time = time.unsqueeze(-1)[*[None]*(len(samples_shape)-1)] # [num_steps] -> [num_steps, 1] -> [...1, num_steps, 1]
        start_samples = start_samples.unsqueeze(-2) # [..., 1, D]

        return self.run_reverse_dynamics_inner(
            start_samples, stimulus, time,
            observations = observations, projection_matrix = projection_matrix, observation_noise_covar = observation_noise_covar,
        )


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    noise_schedule = LinearIncreaseNoiseSchedule(0.2, 0.2, duration = 50)

    score_func = TrueScore(noise_schedule = noise_schedule)

    discretiser = EulerDiscretiser()
    
    diffmodel = ContinuousTimeScoreMatchingDiffusionModel(
        sample_dim=2, noise_schedule=noise_schedule, 
        score_approximator=score_func, discretiser=discretiser
    )

    batch_size = 64
    num_reverse_dynamics_steps = 100

    target_m0 = torch.tensor([-50.0, -10.0])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1)
    target_S0 = torch.tensor([[10.0, -3.0], [-3.0, 4.0]])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1, 1)
    vector_stimulus = (target_m0, target_S0)

    base_samples = torch.randn(batch_size, 2)
    all_reverse_trajectories = diffmodel.run_unconditioned_reverse_dynamics(base_samples, vector_stimulus, num_reverse_dynamics_steps).cpu().numpy()
    example_reverse_trajectories = all_reverse_trajectories[:5]
    end_reverse_samples = all_reverse_trajectories[:,-1,:]

    
    observations = 30.0 + torch.zeros(num_reverse_dynamics_steps, batch_size, 1)  # y
    projection_matrix = torch.tensor([[0.1961, -0.9806]])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1, 1)  # A
    observation_noise_covar = torch.tensor([[1.0, 0.0], [0.0, 1.0]])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1, 1)

    base_samples = torch.randn(batch_size, 2)
    all_conditioned_reverse_trajectories = diffmodel.run_conditioned_reverse_dynamics(
        base_samples, vector_stimulus, num_reverse_dynamics_steps, observations, projection_matrix, observation_noise_covar
    ).cpu().numpy()
    example_conditioned_reverse_trajectories = all_conditioned_reverse_trajectories[:5]
    end_conditioned_reverse_samples = all_conditioned_reverse_trajectories[:,-1,:]
    
    real_samples = torch.randn(batch_size, 2) + target_m0[0,0]      # Not quite real samples!
    all_forward_trajectories = diffmodel.run_forward_dynamics(real_samples, 100).cpu().numpy()
    example_forward_trajectories = all_forward_trajectories[:5]
    end_forward_samples = all_forward_trajectories[:,-1,:]

    fig, axes = plt.subplots(1, 4, figsize = (20, 5))
    scaling_factor_time, scaling_factor = noise_schedule.summarise_noising_factor(100)
    axes[0].plot(scaling_factor_time.cpu().numpy(), scaling_factor.cpu().numpy())
    axes[0].set_ylim(0)

    axes[1].scatter(*end_reverse_samples.T)
    for example_reverse_trajectory in example_reverse_trajectories:
        axes[1].plot(*example_reverse_trajectory.T, alpha = 0.4)
    axes[1].set_aspect(1.0)

    axes[2].scatter(*end_conditioned_reverse_samples.T)
    for example_conditioned_reverse_trajectory in example_conditioned_reverse_trajectories:
        axes[2].plot(*example_conditioned_reverse_trajectory.T, alpha = 0.4)
    axes[2].set_aspect(1.0)

    axes[3].scatter(*end_forward_samples.T)
    for example_forward_trajectory in example_forward_trajectories:
        axes[3].plot(*example_forward_trajectory.T, alpha = 0.4)
    axes[3].set_aspect(1.0)
    
    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/schedule.png')
