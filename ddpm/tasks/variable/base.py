import numpy as np

import torch
from torch import Tensor as _T

import random
from typing import Dict, Set, List
from abc import ABC, abstractmethod

from matplotlib.pyplot import Axes
from matplotlib import pyplot as plt

from purias_utils.multiitem_working_memory.util.circle_utils import (
    generate_circular_feature_list,
    polar2cart,
    rectify_angles
)


# def generate_clifford_torus_directions(ddpm_module: DDPMReverseProcessBase, N_items: int = 3):
#     assert ddpm_module.linking_matrix_kernel.shape[1] - ddpm_module.linking_matrix_kernel.shape[0] == 2
#     assert N_items * 4 <= ddpm_module.linking_matrix_kernel.shape[1]
#     return {
#         'probe_directions': ddpm_module.linking_matrix_kernel[:2 * N_items],
#         'report_directions': ddpm_module.linking_matrix_kernel[2 * N_items:4 * N_items],
#     }



def generate_stimulus_features(N_items: int, batch_size: int, excl_max_items: float) -> Dict[str, _T]:
    "all are [batch_size, num_items, 1/2]"
    all_probe_features, all_report_features = [], []
    for bs in range(batch_size):
        all_probe_features.append(generate_circular_feature_list(N_items, torch.pi / excl_max_items))
        all_report_features.append(
            generate_circular_feature_list(N_items, torch.pi / excl_max_items)
        )
    all_probe_features = torch.tensor(all_probe_features)
    all_report_features = torch.tensor(all_report_features)
    return {
        "probe_features": all_probe_features,
        "report_features": all_report_features,
        "probe_features_cart": torch.stack(polar2cart(1.0, all_probe_features), -1),
        "report_features_cart": torch.stack(polar2cart(1.0, all_report_features), -1),
    }


# def generate_stimulus_features_with_gaps(N_items: int, N_gaps: int, batch_size: int) -> Dict[str, _T]:
#     "all are [batch_size, num_items, 1/2]"
#     all_probe_features, all_report_features = [], []
#     for bs in range(batch_size):
#         gap_locations = ...
#         new_probe_features = generate_circular_feature_list(N_items, torch.pi / 4)
#         new_probe_features_with_gaps = ...
#         new_report_features = generate_circular_feature_list(N_items, torch.pi / 4)
#         new_report_features_with_gaps = ...
#         import pdb; pdb.set_trace(header = 'implement gaps with NaNs')
#         all_probe_features.append(new_probe_features_with_gaps)
#         all_report_features.append(new_report_features_with_gaps)
#     all_probe_features = torch.tensor(all_probe_features)
#     all_report_features = torch.tensor(all_report_features)
#     all_probe_features_cart = polar2cart(1.0, all_probe_features)
#     all_report_features_cart = polar2cart(1.0, all_report_features)
#     all_probe_features_cart = torch.nan_to_num(all_probe_features_cart, nan = 0.0)
#     all_report_features_cart = torch.nan_to_num(all_report_features_cart, nan = 0.0)
#     return {
#         "probe_features": all_probe_features,
#         "report_features": all_report_features,
#         "probe_features_cart": all_probe_features_cart,
#         "report_features_cart": all_report_features_cart,
#     }




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
    def generate_variable_dict(self, batch_size: int, *args, **kwargs) -> Dict[str, _T]:
        raise NotImplementedError

    @abstractmethod
    def display_task_variables(
        self, task_variable_information: Dict[str, _T], *axes: Axes
    ) -> None:
        raise NotImplementedError


class StandardCartesianWMTaskVariableGenerator(TaskVariableGenerator, ABC):
    """
    Just generates a set of item features, N given at init and D = 2
    Also generate a probability of swapping to each of these items, which is processed downstream
    """

    task_variable_keys = {
        #"probe_features",
        "report_features",
        "probe_features_cart",
        "report_features_cart",
        "swap_probabilities",
    }

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self.min_sep = 2 * num_items + 0.1

    @abstractmethod
    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        raise NotImplementedError

    def generate_variable_dict(self, batch_size: int) -> Dict[str, _T]:
        ret = generate_stimulus_features(self.num_items, batch_size, self.min_sep)
        probability_vector_task_variables = self.generate_probability_vectors(ret)
        ret.update(probability_vector_task_variables)
        return ret


class FixedProvidedSwapProbabilityTaskVariableGenerator(
    StandardCartesianWMTaskVariableGenerator
):
    """
    In the absolute simplest case, this probability vector is constant between trials
    We put this here purely for convinience -- the network will not learn anything from this pmf input
    """

    def __init__(self, num_items: int, probability_vector: List[float]) -> None:
        assert sum(probability_vector) == 1.0 and len(probability_vector) == num_items
        self.probability_vector = torch.tensor(probability_vector)
        super().__init__(num_items)

    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        batch_size = variable_dict["report_features"].shape[0]
        assert tuple(variable_dict["report_features"].shape) == (
            batch_size,
            self.num_items,
        )
        return {
            "swap_probabilities": self.probability_vector.unsqueeze(0).expand(
                batch_size, 1
            )
        }

    def display_task_variables(
        self, task_variable_information: Dict[str, _T], *axes: Axes, batch_idx=0
    ) -> None:
        axes[0].set_title("Report feature values with probability of swapping to item")
        axes[0].add_patch(plt.Circle((0, 0), 1.0, color="red", fill=False))
        axes[0].scatter(
            *task_variable_information["report_features_cart"][batch_idx].T, s=50
        )
        for i, prob in enumerate(
            task_variable_information["swap_probabilities"][batch_idx]
        ):
            prob = round(prob.item(), 3)
            axes[0].annotate(
                prob,
                (
                    task_variable_information["report_features_cart"][batch_idx, i, 0],
                    task_variable_information["report_features_cart"][batch_idx, i, 1],
                ),
            )


class ZeroTemperatureSwapProbabilityTaskVariableGenerator(
    StandardCartesianWMTaskVariableGenerator
):
    """
    In another simple case, each batch item is dedicated to swapping to one item at a time
    """

    def __init__(self, num_items: int) -> None:
        super().__init__(num_items)
        self.task_variable_keys = self.task_variable_keys.union({"cued_item_idx"})

    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        """
        selected_item of shape [batch]
        probability_vector of shape [batch, num_items]
        """
        batch_size = variable_dict["report_features"].shape[0]
        assert tuple(variable_dict["report_features"].shape) == (
            batch_size,
            self.num_items,
        )
        probability_vector = torch.zeros(batch_size, self.num_items)
        selected_item = torch.randint(0, self.num_items, (batch_size,))
        probability_vector[range(batch_size), selected_item] = 1.0
        return {
            "swap_probabilities": probability_vector,
            "cued_item_idx": selected_item,
        }

    def display_task_variables(
        self, task_variable_information: Dict[str, _T], *axes: Axes
    ) -> None:
        assert len(axes) >= 1
        axes[0].set_title("Report feature values\nwith probability of swapping to item")
        axes[0].add_patch(plt.Circle((0, 0), 1.0, color="red", fill=False))
        axes[0].scatter(*task_variable_information["report_features_cart"][0].T, s=50)
        for i, prob in enumerate(task_variable_information["swap_probabilities"][0]):
            prob = round(prob.item(), 3)
            axes[0].annotate(
                prob,
                (
                    task_variable_information["report_features_cart"][0, i, 0],
                    task_variable_information["report_features_cart"][0, i, 1],
                ),
            )
            if i == task_variable_information["cued_item_idx"][0]:
                axes[0].add_patch(
                    plt.Circle(
                        task_variable_information["report_features_cart"][
                            0, i
                        ].tolist(),
                        0.1,
                        color="green",
                        fill=False,
                    )
                )

        axes[1].set_title("Probe feature values\nwith probability of swapping to item")
        axes[1].add_patch(plt.Circle((0, 0), 1.0, color="red", fill=False))
        axes[1].scatter(*task_variable_information["probe_features_cart"][0].T, s=50)
        for i, prob in enumerate(task_variable_information["swap_probabilities"][0]):
            prob = round(prob.item(), 3)
            axes[1].annotate(
                prob,
                (
                    task_variable_information["probe_features_cart"][0, i, 0],
                    task_variable_information["probe_features_cart"][0, i, 1],
                ),
            )
            if i == task_variable_information["cued_item_idx"][0]:
                axes[1].add_patch(
                    plt.Circle(
                        task_variable_information["probe_features_cart"][0, i].tolist(),
                        0.1,
                        color="green",
                        fill=False,
                    )
                )


class SpikeAndSlabSwapProbabilityTaskVariableGenerator(
    ZeroTemperatureSwapProbabilityTaskVariableGenerator
):
    """
    In the next case up, cued_item_idx has probability p, and all other items have probability (1-p)/(N-1)
    """

    def __init__(
        self,
        num_items: int,
        correct_probability: float,
        stimulus_exposure_duration: int,
        pre_index_delay_duration: int | List[int],
        index_duration: int,
        post_index_delay_duration: int,
    ) -> None:
        super().__init__(num_items)
        self.task_variable_keys = self.task_variable_keys.union(
            {"cued_item_idx", "prep_epoch_durations", "diffusion_epoch_durations"}
        )
        assert 0.0 <= correct_probability <= 1.0
        self.correct_probability = correct_probability
        self.prep_epoch_durations = [stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration]
        self.diffusion_epoch_durations = [None] # this will just be taken as the full duration by the ddpm

    def generate_prep_epoch_durations(self):
        res = []
        for dur in self.prep_epoch_durations:
            if isinstance(dur, int):
                res.append(dur)
            elif isinstance(dur, list):
                res.append(random.randint(*dur))
            else:
                raise TypeError(dur)
        return res

    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        """
        selected_item of shape [batch]
        probability_vector of shape [batch, num_items]
        """
        batch_size = variable_dict["report_features"].shape[0]
        assert tuple(variable_dict["report_features"].shape) == (
            batch_size,
            self.num_items,
        )
        selected_item = torch.randint(0, self.num_items, (batch_size,))
        selected_item_mask = torch.zeros(batch_size, self.num_items).bool()
        selected_item_mask[range(batch_size), selected_item] = True
        probability_vector = torch.zeros(batch_size, self.num_items)
        probability_vector[selected_item_mask] = self.correct_probability
        probability_vector[~selected_item_mask] = (1.0 - self.correct_probability) / (
            self.num_items - 1
        )
        # assert set(probability_vector.sum(-1).unique().tolist()) == {1.0}
        return {
            "swap_probabilities": probability_vector,
            "cued_item_idx": selected_item,
            "prep_epoch_durations": self.generate_prep_epoch_durations(),
            "diffusion_epoch_durations": self.diffusion_epoch_durations
        }


class ProbeDistanceProbabilityTaskVariableGenerator(
    ZeroTemperatureSwapProbabilityTaskVariableGenerator
):
    """
    Next, we have a swap function that depends on the angular distance in the *probe* dimension only

    f(x = delta probe) = -0.5x^2/(0.5 swap_function_width + eps)
    The probability of swapping is softmax(f(x))
        NB: x in radians!

    Setting swap_function_width = 0 gives the same as SpikeAndSlabSwapProbabilityTaskVariableGenerator
    """
    def __init__(
        self,
        num_items: int,
        swap_function_width: float,
        stimulus_exposure_duration: int,
        pre_index_delay_duration: int | List[int],
        index_duration: int,
        post_index_delay_duration: int,
    ) -> None:
        super().__init__(num_items)
        self.task_variable_keys = self.task_variable_keys.union(
            {"cued_item_idx", "prep_epoch_durations", "diffusion_epoch_durations"}
        )
        self.swap_function_width = swap_function_width
        self.prep_epoch_durations = [stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration]
        self.diffusion_epoch_durations = [None] # this will just be taken as the full duration by the ddpm

    def generate_prep_epoch_durations(self):
        res = []
        for dur in self.prep_epoch_durations:
            if isinstance(dur, int):
                res.append(dur)
            elif isinstance(dur, list):
                res.append(random.randint(*dur))
            else:
                raise TypeError(dur)
        return res

    def generate_probability_vectors(self, variable_dict: Dict[str, _T]) -> _T:
        """
        selected_item of shape [batch]
        probability_vector of shape [batch, num_items]
        """
        batch_size = variable_dict["report_features"].shape[0]
        assert tuple(variable_dict["report_features"].shape) == (
            batch_size,
            self.num_items,
        )
        selected_item = torch.randint(0, self.num_items, (batch_size,))
        selected_item_mask = torch.zeros(batch_size, self.num_items).bool()
        selected_item_mask[range(batch_size), selected_item] = True


        cued_probe = variable_dict["probe_features"][range(batch_size), selected_item].unsqueeze(-1)
        cued_probe_sq_distance = rectify_angles(variable_dict["probe_features"] - cued_probe).square()
        swap_func = -0.5 * (cued_probe_sq_distance / (self.swap_function_width + 2e-5))
        probability_vector = swap_func.softmax(-1)

        # assert set(probability_vector.sum(-1).unique().tolist()) == {1.0}
        return {
            "swap_probabilities": probability_vector,
            "cued_item_idx": selected_item,
            "prep_epoch_durations": self.generate_prep_epoch_durations(),
            "diffusion_epoch_durations": self.diffusion_epoch_durations
        }
