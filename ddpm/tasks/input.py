import torch
from torch import Tensor as _T

from abc import ABC, abstractmethod
from typing import Dict, Set, List


class InputGenerator(ABC):
    """
    Can generate inputs as tabular data or as image inputs

    Inputs are tensor with shape [... self.input_shape]
    
    XXX images only really works with time varying case!
    """
    input_shape: List[int]
    required_task_variable_keys: Set[str]

    @abstractmethod
    def generate_network_inputs(self, variable_dict: Dict[_T]) -> _T:
        raise NotImplementedError


class ProvidedSwapProbabilityInputGenerator(InputGenerator, ABC):
    """
    Simplest task - directly tells the network how often to swap to each item
    Therefore only provides the report dimension as cartesian --> input size of [..., N * 2 + N]
    """
    
    required_task_variable_keys = {'report_features_cart', 'swap_probabilities'}

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self.input_shape = (num_items * 2 + num_items,)

    def generate_network_inputs(self, variable_dict: Dict[_T]) -> _T:
        import pdb; pdb.set_trace(header = 'validate set size!')
        import pdb; pdb.set_trace(header = 'join cartesian features with probability in correct order!')
        return report_features_and_pmfs