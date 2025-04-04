import torch
from torch import Tensor as _T

from typing import Dict, Tuple


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

    def __init__(self, duration: float) -> None:
        super().__init__()
        self.duration = duration

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
        x0 of shape [..., D]
        time of shape [T, ...]

        outputs of shape:
            noising_factor          [T, ...]
            epsilon                 [T, ..., D]
            x_t                     [T, ..., D]
            conditional_score       [T, ..., D]
        """
        nt, *extra_dims = time.shape
        assert list(x0.shape[:-1]) == extra_dims
        sample_dim = x0.shape[-1]

        noising_factor = self.noising_factor(time = time).unsqueeze(-1)                     # [T, ..., 1]
        epsilon = torch.randn(nt, *extra_dims, sample_dim, device = x0.device)              # [T, ..., D]
        xt = (noising_factor * x0) + ((1.0 - noising_factor.square()).sqrt() * epsilon)     # [T, ..., D]
        conditional_score = - epsilon / (1.0 - noising_factor.square()) # XXX: Check before training!   # [T, ..., D]
        return {
            'noising_factor': noising_factor.squeeze(-1),
            'epsilon': epsilon,
            'x_t': xt,
            'conditional_score': conditional_score
        }

    @torch.no_grad()
    def marginal_moments_gaussian_gt_distribution(self, m_x0: _T, S_x0: _T, time: _T) -> Dict[str, _T]:
        """
        Integrate p(x_0) = N(m_x0, S_x0) out of conditional moments, see class docstring
        m_x0 of shape [T, ..., d]
        S_x0 of shape [T, ..., d, d]
        time of shape [T, ...]

        Keep T at start of target distribution in case we want to change it over time!

        TODO: make this multimodal?
        """
        noising_factor = self.noising_factor(time = time)           # [T, ...]

        reshaped_noising_factor = noising_factor[...,None,None]     # [T, ..., 1, 1]
        iden = torch.eye(S_x0.shape[-1])[*[None]*(len(S_x0.shape) - 2)].to(S_x0.device)
        extra_var = (1.0 - reshaped_noising_factor.square()) * iden     # [T, ..., d, d]
        S_xt = (reshaped_noising_factor.square() * S_x0) + extra_var
        return {
            'm_xt': m_x0 * noising_factor.unsqueeze(-1),
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

    def random_noise(self, x0: _T, num_timepoints: int) -> Dict[str, _T]:
        """
        x0 of shape [..., D] XXX allow structured input
        output of shape [..., num_timepoints, D]
        """
        timepoints = torch.rand(num_timepoints) * self.duration
        time = timepoints[:, *[None]*len(x0.shape[:-1])].repeat(1, *x0.shape[:-1])
        return time, self.noise_and_conditional_score(x0=x0, time=time)


class ConstantTimeNoiseSchedule(ContinuousTimeNoiseSchedule):

    def __init__(self, noise_level: float, duration: float = 10.0) -> None:
        super().__init__(duration)
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
        super().__init__(duration)
        self.start_noise_level = start_noise_level    # at time 0
        self.end_noise_level = end_noise_level  # at time 1
        self.slope = end_noise_level - start_noise_level

        smallest_scaling_factor = self.summarise_noising_factor()[1][-1]
        assert 0.01 > smallest_scaling_factor > 0.0001, smallest_scaling_factor.item()
    
    def beta(self, time: _T) -> _T:
        return self.start_noise_level + self.slope * time

    def int_t_beta(self, time: _T) -> _T:
        return (
            self.start_noise_level * time
            + 0.5 * self.slope * time.square()
        )


class CosineNoiseSchedule(ContinuousTimeNoiseSchedule):

    def __init__(self, mag: float, duration: float) -> None:
        super().__init__(duration)
        self.mag = mag
        self.factor = torch.pi / 2 / self.duration

        smallest_scaling_factor = self.summarise_noising_factor()[1][-1]
        assert 0.01 > smallest_scaling_factor > 0.0001, smallest_scaling_factor.item()

    def beta(self, time: _T) -> _T:
        return 1.0 - self.mag * (self.factor * time).cos()
    
    def int_t_beta(self, time: _T) -> _T:
        return time - ((self.mag / self.factor) * (self.factor * time).sin())
