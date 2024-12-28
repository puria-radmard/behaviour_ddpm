import numpy as np

import torch
from torch import Tensor as _T

import random
from typing import Dict, Set, List
from abc import ABC, abstractmethod

from matplotlib.pyplot import Axes
from matplotlib import pyplot as plt

from purias_utils.multiitem_working_memory.util.circle_utils import generate_circular_feature_list, polar2cart


# def generate_clifford_torus_directions(ddpm_module: DDPMReverseProcessBase, N_items: int = 3):
#     assert ddpm_module.linking_matrix_kernel.shape[1] - ddpm_module.linking_matrix_kernel.shape[0] == 2
#     assert N_items * 4 <= ddpm_module.linking_matrix_kernel.shape[1]
#     return {
#         'probe_directions': ddpm_module.linking_matrix_kernel[:2 * N_items],
#         'report_directions': ddpm_module.linking_matrix_kernel[2 * N_items:4 * N_items],
#     }


def generate_stimulus_features(N_items: int) -> Dict[str, _T]:
    probe_features = generate_circular_feature_list(N_items, torch.pi / 4)
    report_features = generate_circular_feature_list(N_items, torch.pi / 4)
    return {
        'probe_features': probe_features,
        'report_features': report_features,
        'probe_features_cart': torch.tensor(np.stack(polar2cart(1.0, probe_features), -1)),
        'report_features_cart': torch.tensor(np.stack(polar2cart(1.0, report_features), -1)),
    }


class TaskVariableGenerator(ABC):
    """
    Must include enough information to generate both the network inputs and the example samples without any extra processing (besides data structure processing)
    Output as a dictionary with keys self.task_variable_keys
    
    TODO: May produce a 'final_mean' variable alongside it also, which is used for the doubly conditioned case
    TODO: extend to time varying case!
    """
    task_variable_keys: Set[str]

    task_metadata = {}

    @abstractmethod
    def generate_variable_dict(self, *args, **kwargs) -> Dict[str, _T]:
        raise NotImplementedError

    @abstractmethod
    def display_task_variables(self, task_variable_information: Dict[str, _T], *axes: Axes) -> None:
        raise NotImplementedError



class StandardCartesianWMTaskVariableGenerator(TaskVariableGenerator, ABC):
    """
    Just generates a set of item features, N given at init and D = 2
    Also generate a probability of swapping to each of these items, which is processed downstream
    """

    task_variable_keys = {'probe_features', 'report_features', 'probe_features_cart', 'report_features_cart', 'swap_probabilities'}

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items

    @abstractmethod
    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        raise NotImplementedError

    def generate_variable_dict(self) -> Dict[str, _T]:
        ret = generate_stimulus_features(self.num_items)
        probability_vector_task_variables = self.generate_probability_vectors(ret)
        ret.update(probability_vector_task_variables)
        return ret



class FixedProvidedSwapProbabilityTaskVariableGenerator(StandardCartesianWMTaskVariableGenerator):
    """
    In the absolute simplest case, this probability vector is constant between trials
    We put this here purely for convinience -- the network will not learn anything from this pmf input
    """
    def __init__(self, num_items: int, probability_vector: List[float]) -> None:
        assert sum(probability_vector) == 1.0 and len(probability_vector) == num_items
        self.probability_vector = torch.tensor(probability_vector)
        super().__init__(num_items)

    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        assert len(variable_dict['report_features']) == self.num_items
        return {'swap_probabilities': self.probability_vector}

    def display_task_variables(self, task_variable_information: Dict[str, _T], *axes: Axes) -> None:
        assert len(axes) >= 1
        axes[0].set_title('Report feature values with probability of swapping to item')
        axes[0].add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))
        axes[0].scatter(*task_variable_information['report_features_cart'].T, s = 50)
        for i, prob in enumerate(task_variable_information['swap_probabilities']):
            prob = round(prob.item(), 3)
            axes[0].annotate(prob, (task_variable_information['report_features_cart'][i,0], task_variable_information['report_features_cart'][i,1]))



class ZeroTemperatureSwapProbabilityTaskVariableGenerator(StandardCartesianWMTaskVariableGenerator):
    """
    In another simple case, each trial is dedicated to swapping to one item at a time
    """
    def __init__(self, num_items: int) -> None:
        super().__init__(num_items)
        self.task_variable_keys = self.task_variable_keys.union({'cued_item_idx'})

    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        assert len(variable_dict['report_features']) == self.num_items
        probability_vector = torch.zeros(self.num_items)
        selected_item = random.randint(0, self.num_items - 1)
        probability_vector[selected_item] = 1.0
        return {'swap_probabilities': probability_vector, 'cued_item_idx': selected_item}
    
    def display_task_variables(self, task_variable_information: Dict[str, _T], *axes: Axes) -> None:
        assert len(axes) >= 1
        axes[0].set_title('Report feature values\nwith probability of swapping to item')
        axes[0].add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))
        axes[0].scatter(*task_variable_information['report_features_cart'].T, s = 50)
        for i, prob in enumerate(task_variable_information['swap_probabilities']):
            prob = round(prob.item(), 3)
            axes[0].annotate(prob, (task_variable_information['report_features_cart'][i,0], task_variable_information['report_features_cart'][i,1]))
            if i == task_variable_information['cued_item_idx']:
                axes[0].add_patch(plt.Circle(task_variable_information['report_features_cart'][i].T.tolist(), 0.1, color='green', fill = False))

        axes[1].set_title('Probe feature values\nwith probability of swapping to item')
        axes[1].add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))
        axes[1].scatter(*task_variable_information['probe_features_cart'].T, s = 50)
        for i, prob in enumerate(task_variable_information['swap_probabilities']):
            prob = round(prob.item(), 3)
            axes[1].annotate(prob, (task_variable_information['probe_features_cart'][i,0], task_variable_information['probe_features_cart'][i,1]))
            if i == task_variable_information['cued_item_idx']:
                axes[1].add_patch(plt.Circle(task_variable_information['probe_features_cart'][i].T.tolist(), 0.1, color='green', fill = False))



