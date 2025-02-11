from sampling_ddpm.ddpm.tasks.variable.base import TaskVariableGenerator, generate_stimulus_features

from typing import Dict, Set, Tuple

import torch
from torch import Tensor as _T

from matplotlib.pyplot import Axes
from matplotlib import pyplot as plt

from abc import abstractmethod

from purias_utils.multiitem_working_memory.util.circle_utils import (
    polar2cart,
    rectify_angles,
)



class InitialisedSampleSpaceTaskVariableGenerator(TaskVariableGenerator):
    """
    To be used with InitialisedSampleSpacePreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess model
        and as part of a InitialisedSampleSpaceMultiEpochDiffusionTask task

    On top of regular task variables, this also produces information used to generate the initial activity
        in the sample space

    This information should be sufficient to allow the InitialisedSampleSpaceMultiEpochDiffusionTask.sample_gen
        to generate initalisation samples without any other information

        See the checks that InitialisedSampleSpaceMultiEpochDiffusionTask.__init__ makes in addition to its
            super().__init__
    """

    pre_prep_variable_keys: Set[str]

    @abstractmethod
    def generate_variable_dict(self, batch_size: int, *args, **kwargs) -> Tuple[Dict[str, _T], Dict[str, _T]]:
        """
        First one has same purpose as before
        Second one is to be fed to sample generator only to generate initialise sample space activity
        """
        raise NotImplementedError




class SequentialCausalInferenceTaskVariableGenerator(InitialisedSampleSpaceTaskVariableGenerator):
    """
    This is like task_variable_probabilistic.StandardCausalInferenceTaskVariableGenerator,
    but this time, the first variable is not given as sensory information but is given to the training
    script as a pre-prep initial condition for the behavioural space

    No memory time: previous_sample_absorb_duration emulates reconfiguration time after previous sampling round
        Stimulus of next item supposed to come in during diffusion round

    In final generation time, stimuli can come in one at a time, with previous_sample_absorb_duration gap between each

    See generate_variable_dict for exact logic
    """

    def __init__(self, previous_sample_absorb_duration: int, min_margin_div_pi: float) -> None:

        self.task_variable_keys = {
            "report_features",
            "report_features_cart",
            "response_locations",
            "response_locations_cart",
            "swap_probabilities",
            "integration_hypothesis"
        }
        
        self.pre_prep_variable_keys = {
            "response_locations",
            "response_locations_cart",
            "swap_probabilities",
            "integration_hypothesis"
        }

        self.prep_epoch_durations = [previous_sample_absorb_duration]
        self.diffusion_epoch_durations = [None]

    def generate_variable_dict(self, batch_size: int) -> Dict[str, _T]:
        """
        Generate two items first, and combine them or keep them seperate as in StandardCausalInferenceTaskVariableGenerator:
            Always have 0.5, 0.5 swap pmf
            Sometimes locations are at report features, sometimes they are both at the average location between the two
        These are used for the pre-prep samples

        Then, a third variable is generated. The following logic is used to generate the final (diffusion) sample distribution:
            1. If the pre-prep samples were unimodel: repeat the same logic with the single mode and the new item
            2. If the pre-prep samples were bimodel: repeat the same logic with the mode which the new item is closest to
        """

        # Generate pre-prep samples with exactly the same logic as in StandardCausalInferenceTaskVariableGenerator
        initial_variables = generate_stimulus_features(2, batch_size)
        initial_items_report_features_distance = rectify_angles(
            initial_variables["report_features"][:, 0] - initial_variables["report_features"][:, 1]
        )
        initial_items_is_close_mask = (
            initial_items_report_features_distance.abs() < self.min_margin_div_pi * torch.pi
        )
        initial_items_average_report_feature = rectify_angles(
            initial_variables["report_features"][initial_items_is_close_mask, 1]
            + 0.5 * initial_items_report_features_distance[initial_items_is_close_mask]
        )
        initial_items_locations = initial_variables["report_features"].clone()
        initial_items_locations[initial_items_is_close_mask] = initial_items_average_report_feature.unsqueeze(1).repeat(1, 2)
        pre_prep_samples_dict = {
            'integration_hypothesis': initial_items_is_close_mask,
            'response_locations': initial_items_locations,
            'response_locations_cart':  torch.stack(polar2cart(1.0, initial_items_locations), -1),
            'swap_probabilities': torch.ones([batch_size, 2]) * 0.5,
        }

        # Introduce a new item, which will determine the target distribution of diffusion
        # Thankfully, the logic above can work the same for both cases, i.e.:
            # regardless of whether the pre-prep samples are unimodal or bimodal, we can 
            # find the miniaml distance between the new variable and pre_prep_samples_dict['response_locations']
            # and reapply the same causal inference logic as above!
        new_variable = generate_stimulus_features(1, batch_size)
        import pdb; pdb.set_trace()

        new_variable_distance = rectify_angles(new_variable - initial_items_locations).abs()
        new_variable_min_distance = new_variable_distance.min(1)
        new_variable_closest_item_idx = new_variable_distance.argmin(1)
        new_variable_is_close_mask = (
            new_variable_min_distance.abs() < self.min_margin_div_pi * torch.pi
        )
        new_items_average_report_feature = rectify_angles(
            initial_items_locations[new_variable_closest_item_idx]
            + 0.5 * new_variable_min_distance[initial_items_is_close_mask]
        )
        new_report_features = ...
        report_features_cart = torch.stack(polar2cart(1.0, new_report_features), -1)
        new_items_locations = ...
        variables_dict = {
            "report_features": new_report_features,
            "report_features_cart": report_features_cart,
            "response_locations": new_items_locations,
            "response_locations_cart": torch.stack(polar2cart(1.0, new_items_locations), -1),
            "swap_probabilities": torch.ones([batch_size, 2]) * 0.5,
            "integration_hypothesis": new_variable_is_close_mask,
            "prep_epoch_durations": self.prep_epoch_durations,
            "diffusion_epoch_durations": self.diffusion_epoch_durations,
        }

        return variables_dict, pre_prep_samples_dict






        ret = generate_stimulus_features(2, batch_size)
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