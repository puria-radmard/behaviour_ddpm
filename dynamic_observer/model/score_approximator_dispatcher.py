import torch
from torch import Tensor as _T

from typing import Tuple

from abc import ABC, abstractmethod

from dynamic_observer.model.noise_schedule import ContinuousTimeNoiseSchedule



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

        reshaped_m_x0, reshaped_S_x0 = stimuli
        assert len(reshaped_m_x0.shape) == 3 and len(reshaped_S_x0.shape) == 4

        self.noise_schedule = noise_schedule
        marginal_moments = self.noise_schedule.marginal_moments_gaussian_gt_distribution(reshaped_m_x0, reshaped_S_x0, t)

        self.m_xt = marginal_moments['m_xt']    # [num steps, ... D]
        self.S_xt = marginal_moments['S_xt']    # [num steps, ... D, D]

    def approximate_score(self, x_t: _T, t_tilde_idx: int, **kwargs):
        m_xt = self.m_xt[t_tilde_idx]
        S_xt = self.S_xt[t_tilde_idx]
        return self.noise_schedule.marginal_score(m_xt, S_xt, x_t)
