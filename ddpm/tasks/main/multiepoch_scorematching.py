from torch import Tensor as _T

from typing import Dict, List, Optional

import numpy as np

from dataclasses import dataclass

from sampling_ddpm.ddpm.tasks.sample.base import ExampleSampleGenerator, SwapSampleInformation
from sampling_ddpm.ddpm.tasks.main.base import DiffusionTask
from sampling_ddpm.ddpm.tasks.variable.base import TaskVariableGenerator
from ddpm.tasks.distribution import DistributionInformation, DistributionInformationGenerator
from sampling_ddpm.ddpm.tasks.input.multiepoch import MultiEpochSensoryGenerator


@dataclass
class ScoreMatchingMultiepochTrialInformation:
    task_variable_information: Dict[str, _T]
    prep_network_inputs: List[_T]
    diffusion_network_inputs: List[_T]
    prep_epoch_durations: List[int]
    diffusion_epoch_durations: List[Optional[int]]
    sample_information: DistributionInformation



class ScoreMatchingMultiEpochDiffusionTask(DiffusionTask):

    def __init__(
        self,
        task_variable_gen: TaskVariableGenerator,
        sensory_gen: MultiEpochSensoryGenerator,
        distribution_gen: DistributionInformationGenerator,
    ) -> None:
        self.task_variable_gen = task_variable_gen
        self.sensory_gen = sensory_gen
        self.distribution_gen = distribution_gen

        self.task_metadata: Dict[str, Dict] = {
            "task_variable_gen_metadata": task_variable_gen.task_metadata,
            "sensory_gen": sensory_gen.task_metadata,
            "distribution_gen": distribution_gen.task_metadata,
        }

        assert sensory_gen.required_task_variable_keys.issubset(
            task_variable_gen.task_variable_keys
        )
        assert distribution_gen.required_task_variable_keys.issubset(
            task_variable_gen.task_variable_keys
        )

    def generate_trial_information(
        self, batch_size: int, num_samples: int, **task_variables_kwargs
    ) -> ScoreMatchingMultiepochTrialInformation:
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
        diffusion_network_inputs = [
            dni.unsqueeze(1).expand(dni.shape[0], num_samples, *dni.shape[1:])
            for dni in diffusion_network_inputs
        ]
        distribution_information = self.distribution_gen.generate_distribution_information(
            num_samples, task_variable_information
        )
        prep_epoch_durations = task_variable_information["prep_epoch_durations"]
        diffusion_epoch_durations = task_variable_information["diffusion_epoch_durations"]
        return ScoreMatchingMultiepochTrialInformation(
            task_variable_information,
            prep_network_inputs,
            diffusion_network_inputs,
            prep_epoch_durations,
            diffusion_epoch_durations,
            sample_information,
        )
