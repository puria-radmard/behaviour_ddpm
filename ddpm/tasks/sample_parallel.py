# 21.01.2025
# A wrapper to make sampling parallelised
# This just means concatenating a bunch of samples from the same source into a larger sample space
# Individual samples are independent, which is fine because they are projected to orthogonal axes in the `cognitive space`

import torch
from torch import Tensor as _T
from numpy import ndarray as _A

from typing import Dict, Tuple, Deque, Optional, Union

from matplotlib.pyplot import Axes
from matplotlib.cm import ScalarMappable

from ddpm.tasks.sample import ExampleSampleGenerator, VectoralEmbeddedExampleSampleGenerator, SwapSampleInformation





class ParallelExampleSampleGenerator(ExampleSampleGenerator):

    def __init__(self, underlying_sample_generator: ExampleSampleGenerator, num_parallel_samples: int) -> None:
        
        assert isinstance(underlying_sample_generator, VectoralEmbeddedExampleSampleGenerator), \
            f"ParallelExampleSampleGenerator is currently only implemented for VectoralEmbeddedExampleSampleGenerator, not for {underlying_sample_generator.__class__.__name__}"
        
        self.underlying_sample_generator = underlying_sample_generator

        self.num_parallel_samples = num_parallel_samples
        self.sample_shape = [self.underlying_sample_generator.sample_shape[0] * num_parallel_samples]

        self.required_task_variable_keys = self.underlying_sample_generator.required_task_variable_keys

        assert not self.underlying_sample_generator.residual_in_behaviour_plane_only
        
        self.task_metadata = self.underlying_sample_generator.task_metadata
    
    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        
        sample_sets = [self.underlying_sample_generator.generate_sample_set(num_samples, variable_dict) for _ in range(self.num_parallel_samples)]
        
        parallelised_selected_cartesian = torch.concat([ss.sample_set for ss in sample_sets], -1)   # [batch, samples, parallel samples * 2]
        parallelised_selected_item_idx = torch.stack([ss.item_indices for ss in sample_sets], -1)   # [batch, samples, parallel samples]

        return SwapSampleInformation(parallelised_selected_cartesian, parallelised_selected_item_idx)

    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T], recent_sample_diagnostics: Deque[_A], axes: Optional[Axes] = None) -> Tuple[_T, Dict[str, _T]]:
        raise NotImplementedError
    
    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes) -> None:
        if isinstance(sample_set, SwapSampleInformation):
            samples = sample_set.sample_set
        else:
            samples = sample_set
        underlying_sample_size = self.underlying_sample_generator.sample_shape[0]   # 2
        for i in range(self.num_parallel_samples):
            self.underlying_sample_generator.display_samples(
                sample_set = samples[...,i * underlying_sample_size: i * underlying_sample_size + 2],
                axes = axes
            )
        pass

    def display_early_x0_pred_timeseries(self, sample_set: _T, axes: Axes, cmap: ScalarMappable) -> None:
        "Too much information!"
        pass
    
    def mse(self, epsilon_hat: _T, epsilon: _T):
        return super(ParallelExampleSampleGenerator, self).mse(
            epsilon_hat, epsilon
        )
