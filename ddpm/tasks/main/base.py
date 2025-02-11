from torch import Tensor as _T

from typing import Dict

import numpy as np

from dataclasses import dataclass

from sampling_ddpm.ddpm.tasks.variable.base import TaskVariableGenerator
from sampling_ddpm.ddpm.tasks.input.base import SensoryGenerator
from sampling_ddpm.ddpm.tasks.sample.base import ExampleSampleGenerator, SwapSampleInformation


@dataclass
class TrialInformation:
    task_variable_information: Dict[str, _T]
    network_inputs: _T
    sample_information: SwapSampleInformation


class DiffusionTask:

    def __init__(
        self,
        task_variable_gen: TaskVariableGenerator,
        sensory_gen: SensoryGenerator,
        sample_gen: ExampleSampleGenerator,
    ) -> None:

        self.task_variable_gen = task_variable_gen
        self.sensory_gen = sensory_gen
        self.sample_gen = sample_gen

        self.task_metadata: Dict[str, Dict] = {
            "task_variable_gen_metadata": task_variable_gen.task_metadata,
            "sensory_gen": sensory_gen.task_metadata,
            "sample_gen": sample_gen.task_metadata,
        }

        assert sensory_gen.required_task_variable_keys.issubset(
            task_variable_gen.task_variable_keys
        )
        assert sample_gen.required_task_variable_keys.issubset(
            task_variable_gen.task_variable_keys
        )

    def generate_trial_information(
        self, batch_size: int, num_samples: int, **task_variables_kwargs
    ) -> TrialInformation:
        """
        task_variable_information of shape [batch, ...]
        sample_information values of shape [batch, samples, ...]
        network_inputs values should be of shape [batch, ...] but for ease downstream we expand to the sample dimension too...
        """
        task_variable_information = self.task_variable_gen.generate_variable_dict(
            batch_size=batch_size, **task_variables_kwargs
        )
        network_inputs = self.sensory_gen.generate_sensory_inputs(
            task_variable_information
        )
        network_inputs = network_inputs.unsqueeze(1).expand(
            batch_size, num_samples, *self.sensory_gen.sensory_shape
        )
        sample_information = self.sample_gen.generate_sample_set(
            num_samples, task_variable_information
        )
        return TrialInformation(
            task_variable_information, network_inputs, sample_information
        )

    def save_metadata(self, path: str):
        np.save(path, self.task_metadata)

    def load_metadata(self, path):
        self.task_metadata: Dict[str, Dict] = np.load(path, allow_pickle=True).item()
