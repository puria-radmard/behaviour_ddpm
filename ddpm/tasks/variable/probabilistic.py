# 17.01.2025
# Made for probabilstic tasks - multisensory integration, causal inference, etc.
# n.b. all tasks are stochastic due to diffusion, this is specifically for tasks that deal with probabilistic quantities

from ddpm.tasks.variable.base import TaskVariableGenerator, generate_stimulus_features
from typing import Dict

import torch
from torch import Tensor as _T

from matplotlib.pyplot import Axes
from matplotlib import pyplot as plt

from purias_utils.multiitem_working_memory.util.circle_utils import (
    polar2cart,
    rectify_angles,
)


class StandardCausalInferenceTaskVariableGenerator(TaskVariableGenerator):
    """
    The most basic causal inference task:
    - two items shown on circle
    - if close, then take their average location (on circle)
    - if far, then bimodal distribution with modes on both

    No mention of uncertainties or magnitudes of each stimulus, which should be added later!

    response_locations is new because we are asking sample generator to generate examples that are not in the
        original report_features_cart set
    """

    task_variable_keys = {
        "report_features",
        "report_features_cart",
        "response_locations",
        "response_locations_cart",
        "swap_probabilities",
        "integration_hypothesis"
    }

    def __init__(self, min_margin_div_pi: float) -> None:
        super().__init__()
        self.min_margin_div_pi = min_margin_div_pi

    def generate_variable_dict(self, batch_size: int) -> Dict[str, _T]:
        """
        Always have 0.5, 0.5 swap pmf
        Sometimes locations are at report features, sometimes they are both at the average location between the two
        """
        ret = generate_stimulus_features(2, batch_size, 7.1)
        report_features_distance = rectify_angles(
            ret["report_features"][:, 0] - ret["report_features"][:, 1]
        )
        is_close_mask = (
            report_features_distance.abs() < self.min_margin_div_pi * torch.pi
        )
        average_report_feature = rectify_angles(
            ret["report_features"][is_close_mask, 1]
            + 0.5 * report_features_distance[is_close_mask]
        )
        locations = ret["report_features"].clone()
        locations[is_close_mask] = average_report_feature.unsqueeze(1).repeat(1, 2)
        ret["response_locations"] = locations
        ret["response_locations_cart"] = torch.stack(polar2cart(1.0, locations), -1)
        ret["swap_probabilities"] = torch.ones([batch_size, 2]) * 0.5
        ret["integration_hypothesis"] = is_close_mask
        ret["report_features_distance"] = report_features_distance
        return ret

    def display_task_variables(
        self, task_variable_information: Dict[str, _T], *axes: Axes, batch_idx=0
    ) -> None:

        axes[0].set_title(
            "Report feature values with probability of providing response"
        )
        axes[0].add_patch(plt.Circle((0, 0), 1.0, color="red", fill=False))
        axes[0].scatter(
            *task_variable_information["report_features_cart"][batch_idx].T,
            s=70,
            label="presented stimuli"
        )
        axes[0].scatter(
            *task_variable_information["response_locations_cart"][batch_idx].T,
            s=50,
            label="mode of response samples"
        )
        axes[0].legend()
        for i, prob in enumerate(
            task_variable_information["swap_probabilities"][batch_idx]
        ):
            prob = round(prob.item(), 3)
            axes[0].annotate(
                prob,
                (
                    task_variable_information["response_locations_cart"][
                        batch_idx, i, 0
                    ],
                    task_variable_information["response_locations_cart"][
                        batch_idx, i, 1
                    ],
                ),
            )





class WeightedCausalInferenceTaskVariableGenerator(StandardCausalInferenceTaskVariableGenerator):
    """
    Same as above, except each stimulus also has a precision.
    
    If the seperation hypothesis is accepted, just combine the modes
    If the integration hypothesis is accepted, Bayesian optimal combination
    """

    task_variable_keys = {
        "report_features",
        "report_features_cart",
        "report_features_prec",
        "response_locations",
        "response_locations_cart",
        "response_locations_prec",
        "swap_probabilities",
        "integration_hypothesis"
    }
    
    def __init__(self, min_margin_div_pi: float) -> None:
        super().__init__(min_margin_div_pi)
    
    def generate_variable_dict(self, batch_size: int) -> Dict[str, _T]:
        ret = super().generate_variable_dict(batch_size)
        report_features_distance = ret["report_features_distance"]
        is_close_mask = ret["integration_hypothesis"]

        average_report_feature = rectify_angles(
            ret["report_features"][is_close_mask, 1]
            + 0.5 * report_features_distance[is_close_mask]
        )

        raise NotImplementedError
