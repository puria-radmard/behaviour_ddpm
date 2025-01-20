from torch import Tensor as _T

from typing import Dict, List

import numpy as np

from dataclasses import dataclass

from ddpm.tasks.sample import ExampleSampleGenerator, SwapSampleInformation
from ddpm.tasks.main import DiffusionTask
from ddpm.tasks.task_variable import TaskVariableGenerator
from ddpm.tasks.multiepoch_input import MultiEpochSensoryGenerator


@dataclass
class MultiepochTrialInformation:
    task_variable_information: Dict[str, _T]
    prep_network_inputs: List[_T]
    diffusion_network_inputs: _T
    prep_epoch_durations: List[int]
    sample_information: SwapSampleInformation


class MultiEpochDiffusionTask(DiffusionTask):

    def __init__(
        self,
        task_variable_gen: TaskVariableGenerator,
        sensory_gen: MultiEpochSensoryGenerator,
        sample_gen: ExampleSampleGenerator,
    ) -> None:
        super().__init__(task_variable_gen, sensory_gen, sample_gen)
        self.sensory_gen: MultiEpochSensoryGenerator  # For typing lateer

    def generate_trial_information(
        self, batch_size: int, num_samples: int, **task_variables_kwargs
    ) -> MultiepochTrialInformation:
        task_variable_information = self.task_variable_gen.generate_variable_dict(
            batch_size=batch_size, **task_variables_kwargs
        )
        prep_network_inputs = self.sensory_gen.generate_prep_sensory_inputs(
            task_variable_information
        )
        diffusion_network_inputs = self.sensory_gen.generate_diffusion_sensory_inputs(
            task_variable_information
        )
        prep_network_inputs = [
            pni.unsqueeze(1).expand(pni.shape[0], num_samples, *pni.shape[1:])
            for pni in prep_network_inputs
        ]
        diffusion_network_inputs = diffusion_network_inputs.unsqueeze(1).expand(
            diffusion_network_inputs.shape[0],
            num_samples,
            *diffusion_network_inputs.shape[1:]
        )
        sample_information = self.sample_gen.generate_sample_set(
            num_samples, task_variable_information
        )
        prep_epoch_durations = task_variable_information["prep_epoch_durations"]
        return MultiepochTrialInformation(
            task_variable_information,
            prep_network_inputs,
            diffusion_network_inputs,
            prep_epoch_durations,
            sample_information,
        )
