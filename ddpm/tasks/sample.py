from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Deque, Optional
from matplotlib.pyplot import Axes

from matplotlib import pyplot as plt

from torch import Tensor as _T
from numpy import ndarray as _A


class ExampleSampleGenerator(ABC):
    """
    Can generate inputs as tabular data or as target samples

    Samples are tensor with shape [num_samples, *self.sample_shape], indexed by 'samples'
    Also output some sample metadata

    Because it contains information about the sample space, it must also be able to process samples,
        alongside the report dimension features provided as stimuli, and determine the rate of swapping to each item.
        Also determine other relevatn diagnostics
    See child class docstrings for specific examples

    Similarly, needs to be able to display (some) example samples
    """
    sample_shape: List[int]
    required_task_variable_keys: Set[str]
    diagnostics_memory = 100

    @abstractmethod
    def generate_sample_set(self, num_samples: int, variable_dict: Dict[_T]) -> Tuple[_T, Dict[_T]]:
        raise NotImplementedError

    @abstractmethod
    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[_T], recent_sample_diagnostics: Deque[_A], axes: Optional[Axes] = None) -> Tuple[_T, Dict[_T]]:
        raise NotImplementedError

    @abstractmethod
    def display_samples(self, sample_set: _T, axes: Axes) -> None:
        raise NotImplementedError


class TabularExampleSampleGenerator(ExampleSampleGenerator):
    """
    Just generates Cartesian samples

    Swapping determined by distance on Cartesian plane
    Only diagnostic is that magnitude of output samples ~= 1
    """
    required_task_variable_keys = {'report_features_cart', 'swap_probabilities'}
    sample_shape = [2]

    def __init__(self) -> None:
        pass

    def generate_sample_set(self, num_samples: int, variable_dict: Dict[_T]) -> Tuple[_T, Dict[_T]]:
        import pdb; pdb.set_trace(header = 'select according to pmf!')
        return selected_cartesian

    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[_T]) -> Tuple[_T, Dict[_T]]:
        import pdb; pdb.set_trace(header = 'get pmf by taking closest to report_feature')
        return closest_rate, {'magnitude': magnitudes}

    def display_samples(self, sample_set: _T, axes: Axes) -> None:
        axes.scatter(sample_set[:, 0], sample_set[:, 1], alpha=0.5, s=1)
        axes.add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))
