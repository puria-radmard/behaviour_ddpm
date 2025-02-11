import torch
from torch import Tensor as _T

from abc import ABC, abstractmethod
from typing import Dict, Set, List


class MultiEpochSensoryGenerator(ABC):
    """
    Can generate sensory inputs as vectoral data or as images

    Outputs are tensor with shape [... self.prep_sensory_shape[i]] or [... self.diffusion_sensory_shapes[i]]

    self.underlying_sensory_shape is used to design the input block

    XXX images only really works with time varying case!
    """

    prep_sensory_shape: List[List[int]]
    diffusion_sensory_shapes: List[List[int]]
    underlying_sensory_shape: List[int]
    required_task_variable_keys: Set[str]

    task_metadata = {}

    @abstractmethod
    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        raise NotImplementedError

    @abstractmethod
    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        raise NotImplementedError


class DelayedIndexCuingSensoryGeneratorWithMemory(MultiEpochSensoryGenerator):
    """
    Prep epoch 1: provide report dimensions as cartestians
    Prep epoch 2: nothing
    Prep epoch 3: provide just the index
    Prep epoch 4: nothing
    Diffusion epoch 1: nothing (but pass as zeros of same shape as prep epoch 1)
    """

    def __init__(self, num_items: int) -> None:
        self.required_task_variable_keys = {"report_features_cart", "cued_item_idx"}
        self.num_items = num_items
        self.underlying_sensory_shape = [num_items * 2]
        self.prep_sensory_shape = [[num_items * 2], [num_items * 2], [1], [num_items * 2]]
        self.diffusion_sensory_shapes = [[num_items * 2]]

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        assert tuple(variable_dict["report_features_cart"].shape) == (
            batch_size,
            self.num_items,
            2,
        )
        flattened_coords = variable_dict["report_features_cart"].reshape(
            batch_size, -1
        )  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        index = variable_dict["cued_item_idx"].unsqueeze(-1)
        empty = torch.zeros(batch_size, self.num_items * 2)
        return [flattened_coords, empty, index, empty]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        return [torch.zeros(batch_size, self.num_items * 2)]



class DelayedProbeCuingSensoryGeneratorWithMemory(MultiEpochSensoryGenerator):
    """
    Prep epoch 1: provide report and probe dimensions as cartestians, zero at end of input vector
    Prep epoch 2: nothing (but pass as zeros of same shape as prep epoch 1)
    Prep epoch 3: provide just the probe cartesians at end of input vector
    Prep epoch 4: nothing (but pass as zeros of same shape as prep epoch 1)
    Diffusion epoch 1: nothing (but pass as zeros of same shape as prep epoch 1)
    """

    def __init__(self, num_items: int) -> None:
        self.required_task_variable_keys = {"report_features_cart", "probe_features_cart", "cued_item_idx"}
        self.num_items = num_items
        self.underlying_sensory_shape = [num_items * 4 + 2]
        self.prep_sensory_shape = [self.underlying_sensory_shape] * 4
        self.diffusion_sensory_shapes = [self.underlying_sensory_shape]

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        assert tuple(variable_dict["report_features_cart"].shape) == tuple(variable_dict["probe_features_cart"].shape) == (
            batch_size,
            self.num_items,
            2,
        )
        flattened_report_coords = variable_dict["report_features_cart"].reshape(
            batch_size, -1
        )  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        flattened_probe_coords = variable_dict["probe_features_cart"].reshape(
            batch_size, -1
        )  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        empty_probe = torch.zeros(batch_size, 2)
        flattened_coords_with_empty = torch.concat([flattened_report_coords, flattened_probe_coords, empty_probe], -1)
        index = variable_dict["cued_item_idx"]
        empty_features = torch.zeros(batch_size, self.num_items * 4)
        chosen_probe_coords = variable_dict["report_features_cart"][range(batch_size), index]
        chosen_probe_coords = torch.concat([empty_features, chosen_probe_coords], -1)
        empty = torch.zeros(batch_size, self.num_items * 4 + 2)
        return [flattened_coords_with_empty, empty, chosen_probe_coords, empty]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        return [torch.zeros(batch_size, self.num_items * 4 + 2)]
