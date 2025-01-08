import torch
from torch import Tensor as _T

from abc import ABC, abstractmethod
from typing import Dict, Set, List


class SensoryGenerator(ABC):
    """
    Can generate sensory inputs as vectoral data or as images

    Outputs are tensor with shape [... self.sensory_shape]
    
    XXX images only really works with time varying case!
    """
    sensory_shape: List[int]
    required_task_variable_keys: Set[str]

    task_metadata = {}

    @abstractmethod
    def generate_sensory_inputs(self, variable_dict: Dict[str, _T]) -> _T:
        raise NotImplementedError


class ProvidedSwapProbabilitySensoryGenerator(SensoryGenerator):
    """
    Simplest task - directly tells the network how often to swap to each item
    Therefore only provides the report dimension as cartesian --> input size of [..., N * 2 + N]
    """
    
    required_task_variable_keys = {'report_features_cart', 'swap_probabilities'}

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self.sensory_shape = (num_items * 2 + num_items,)

    def generate_sensory_inputs(self, variable_dict: Dict[str, _T]) -> _T:
        batch_size = variable_dict['probe_features_cart'].shape[0]
        assert tuple(variable_dict['probe_features_cart'].shape) == (batch_size, self.num_items, 2)
        assert tuple(variable_dict['swap_probabilities'].shape) == (batch_size, self.num_items, )
        flattened_coords = variable_dict['report_features_cart'].reshape(batch_size, -1)  # on each batch row: x1, y1, x2, y2, ...
        report_features_and_pmfs = torch.concat([flattened_coords, variable_dict['swap_probabilities']], 1)
        return report_features_and_pmfs


class IndexCuingSensoryGenerator(SensoryGenerator):
    """
    Second simplest task - give only the recall dimensions, and the item which is cued

    This part of the pipeline has no awareness of swap_probabilities - that needs to be tuned in sample.py
    """

    required_task_variable_keys = {'report_features_cart', 'cued_item_idx'}

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self.sensory_shape = (num_items * 2 + num_items,)
    
    def generate_sensory_inputs(self, variable_dict: Dict[str, _T]) -> _T:
        batch_size = variable_dict['probe_features_cart'].shape[0]
        assert tuple(variable_dict['probe_features_cart'].shape) == (batch_size, self.num_items, 2)
        flattened_coords = variable_dict['report_features_cart'].reshape(batch_size, -1)  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        item_cued_ohe = torch.nn.functional.one_hot(variable_dict['cued_item_idx'], num_classes=self.num_items)     # [batch, num_items]
        report_features_and_index = torch.concat([flattened_coords, item_cued_ohe], -1)
        return report_features_and_index




class ProbeCuingSensoryGenerator(SensoryGenerator):
    """
    Almost the full WM task now - give both dimensions, and 

    This part of the pipeline has no awareness of swap_probabilities - that needs to be tuned in sample.py
    """

    required_task_variable_keys = {'report_features_cart', 'probe_features_cart', 'cued_item_idx'}

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self.sensory_shape = (num_items * 4 + 2,)
    
    def generate_sensory_inputs(self, variable_dict: Dict[str, _T]) -> _T:
        batch_size = variable_dict['probe_features_cart'].shape[0]
        assert tuple(variable_dict['probe_features_cart'].shape) == (batch_size, self.num_items, 2)
        flattened_coords = variable_dict['report_features_cart'].reshape(batch_size, -1)  # on each batch row: x1, y1, x2, y2, ...
        flattened_probe_coords = variable_dict['probe_features_cart'].reshape(batch_size, -1)  # on each batch row: x1, y1, x2, y2, ...
        selected_probe_coords = torch.stack([variable_dict['probe_features_cart'][b,i] for b, i in enumerate(variable_dict['cued_item_idx'])])
        report_features_and_index = torch.concat([flattened_coords, flattened_probe_coords, selected_probe_coords], 1)
        return report_features_and_index

