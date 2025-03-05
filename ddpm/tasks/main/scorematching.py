from torch import Tensor as _T

from typing import Dict, List, Optional

import numpy as np

from dataclasses import dataclass

from ddpm.tasks.main.base import DiffusionTask
from ddpm.tasks.variable.base import TaskVariableGenerator
from ddpm.tasks.distribution import DistributionInformation, DistributionInformationGenerator
from ddpm.tasks.input.multiepoch import MultiEpochSensoryGenerator


@dataclass
class ScoreMatchingTrialInformation:
    task_variable_information: Dict[str, _T]
    network_inputs: _T
    distribution_information: DistributionInformation



class ScoreMatchingDiffusionTask(DiffusionTask):

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
    ) -> ScoreMatchingTrialInformation:

        task_variable_information = self.task_variable_gen.generate_variable_dict(
            batch_size=batch_size, **task_variables_kwargs
        )
        network_inputs = self.sensory_gen.generate_sensory_inputs(
            task_variable_information
        )
        network_inputs = network_inputs.unsqueeze(1).expand(
            batch_size, num_samples, *self.sensory_gen.sensory_shape
        )
        distribution_information = self.distribution_gen.generate_distribution_information(
            task_variable_information
        )
        return ScoreMatchingTrialInformation(
            task_variable_information, network_inputs, distribution_information
        )

