from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Deque, Optional, Union
from matplotlib.pyplot import Axes

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

import math
import torch
import numpy as np

from torch import Tensor as _T
from numpy import ndarray as _A



class DistributionInformation(ABC):
    """
    Roles:
        - Fully describes the target distribution
        - Given a_t and b_t (see model schedules), can integrate out target distribution to get evolving marginals
            - Can do this both for one a_t/b_t at a time (for teacher forcing)...
            - ... or for the full schedule (for target generation)
        - 
    """
    
    def __init__(self, sample_shape: List[int], batch_shape: List[int]) -> None:
        self.sample_shape = sample_shape
        self.batch_shape = batch_shape
        self.num_extra_dims = len(self.batch_shape)

    @abstractmethod
    def calculate_score(self, x_t: _T, a_t: _T, b_t: _T) -> _T:
        """
        x_t: of shape [..., T, <sample_shape>]
        a_t, b_t: of shape [T] only

        returns of shape [..., T, <sample_shape>]
        """
        raise NotImplementedError

    @abstractmethod
    def display_final_score(
        self, axes: Axes, plotting_kwargs: Optional[Dict] = {}
    ) -> None:
        raise NotImplementedError



class DiagonalGaussianMixtureDistributionInformation(DistributionInformation):
    """
    weights of shape [..., num_modes]
    means of shape [..., num_modes, sample_size]
    std2s of shape [..., num_modes, sample_size]
    """
    def __init__(self, weights: _T, means: _T, std2s: _T, sample_size: int) -> None:
        *batch_shape, self.num_modes, dim = std2s.shape
        super().__init__([sample_size], batch_shape)
        assert dim == sample_size
        assert means.shape == std2s.shape and std2s.shape[:-1] == weights.shape
        self.pi = weights
        self.mu = means
        self.s2 = std2s

        assert self.batch_shape == [self.batch_shape[0], 1], \
            "A lot of things currently assuem that DiagonalGaussianMixtureDistributionInformation.batch_shape is [batch size, 1]"

        self.gaussian_normalisation = (
            torch.pow(2 * torch.pi, torch.tensor(self.sample_shape[0] / 2))
            * torch.sqrt(self.s2.prod(-1))
        ).unsqueeze(-2) # [<B>, 1, M], <B> = batch shape
    
    def calculate_score(self, x_t: _T, a_t: _T, b_t: _T) -> _T:
        """
        <B> = batch shape
        T = num timesteps
        M = num modes
        D = dimensionality of data

        x: reverse process shaped [<B>, T, D]
        mu: mode means shaped [<B>, M, D]
        s2: mode std in each direction directions, [<B>, M, D]
        pi: weights sized [<B>, M]
        b_t, a_t: [T]

        return: [<B>, T, D]
        """
        timesteps = a_t.shape[0]
        assert list(a_t.shape) == list(b_t.shape) == [timesteps]
        assert list(x_t.shape) == [*x_t.shape[:self.num_extra_dims], timesteps, self.sample_shape[0]]

        # useful stuff
        reshaped_a_t = a_t[*[None]*self.num_extra_dims,:,None,None]      # [<1*B>, T, 1, 1]
        reshaped_b_t = b_t[*[None]*self.num_extra_dims,:,None,None]      # [<1*B>, T, 1, 1]
        diffused_means = reshaped_a_t * self.mu.unsqueeze(-3)       # [<B>, T, M, D]
        diffused_std2s = reshaped_b_t + (torch.pow(reshaped_a_t, 2.0) * self.s2.unsqueeze(-3))   # [<B>, T, M, D]
        error_from_mode = x_t.unsqueeze(-2) - diffused_means        # [<B>, T, M, D]
        score_linear_weighter = - 0.5 * error_from_mode / diffused_std2s     # [<B>, T, M, D]

        # denominator = pdf [<B>, T]
        gaussian_exponent = (score_linear_weighter * error_from_mode).sum(-1)   # [<B>, T, M]
        scaled_pdf_by_mode = gaussian_exponent.exp() * self.pi.unsqueeze(-2) / self.gaussian_normalisation     # [<B>, T, M]
        pdf = scaled_pdf_by_mode.sum(-1)

        # numerator = grad [<B>, T, M, D] -> [<B>, T, D]
        pdf_grad = 2 * (score_linear_weighter * scaled_pdf_by_mode.unsqueeze(-1)).sum(-2)
        
        score = pdf_grad / pdf.unsqueeze(-1)

        return score

    def display_final_score(
        self, left: float, right: float, top: float, bottom: float,
        axes: Axes, plotting_kwargs: Optional[Dict] = {}
    ) -> None:
        assert self.sample_shape == [2]
        x, y = torch.meshgrid(torch.linspace(left, right, 20), torch.linspace(bottom, top, 20)) 
        x_t = torch.stack([x, y], -1)
        x_t = x_t.reshape(-1, 1, 1, 2)
        score = self.calculate_score(
            x_t = x_t, a_t = torch.tensor([1.0]), b_t = torch.tensor([0.0])
        )[:,0,0,:].reshape(20, 20, 2)
        u, v = score.permute(2, 0, 1)
        axes.quiver(x, y, u, v, **plotting_kwargs) 


class DiagonalGaussianOnCircleMixtureDistributionInformation(DiagonalGaussianMixtureDistributionInformation):

    def __init__(self, weights: _T, angular_locations: _T, sample_size: int) -> None:
        super().__init__(weights, means, std2s, 2)


class DistributionInformationGenerator(ABC):
    """
    A class for when the final target distribution is given as closed form,
        rather than by a set of samples like in sample.SwapSampleInformation

    We get rid of generate_sample_set -> SwapSampleInformation
        as we no longer generate samples (but can still display_samples).
    Instead, we generate an instance of DistributionInformation using generate_distribution_information
        This is an object that is dispatched to the model class
        The role is described in the ScoreMatchingHelper docstring
    """
    sample_shape: List[int]
    required_task_variable_keys: Set[str]

    @abstractmethod
    def generate_distribution_information(
        self, 
    ) -> DistributionInformation:
        raise NotImplementedError

    @abstractmethod
    def display_samples(
        self, sample_set: _T, axes: Axes
    ) -> None:
        raise NotImplementedError
    
    def score_mse(self, score_hat: _T, score: _T):
        """
        Both coming in shape [..., T, <sample shape>]
        """
        assert score.shape == score_hat.shape
        batch_size, T = score_hat.shape[:2]
        *extra_dims, T = score_hat.shape[: -len(self.sample_shape)]
        mse = (
            torch.square(score_hat - score).reshape(*extra_dims, T, -1).mean(-1)
        )  # [..., T]
        # scaled_mse = self.mse_scaler_schedule[*[None]*extra_dims] * mse[...,1:] # [..., T-1]
        return mse


