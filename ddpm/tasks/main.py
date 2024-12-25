from torch import Tensor as _T

from typing import Dict

from dataclasses import dataclass

from ddpm.tasks.task_variable import TaskVariableGenerator
from ddpm.tasks.input import InputGenerator
from ddpm.tasks.sample import ExampleSampleGenerator, SwapSampleInformation


@dataclass
class TrialInformation:
    task_variable_information: Dict[str, _T]
    network_inputs: _T
    sample_information: SwapSampleInformation


class WMDiffusionTask:

    def __init__(self, task_variable_gen: TaskVariableGenerator, input_gen: InputGenerator, sample_gen: ExampleSampleGenerator) -> None:
        
        self.task_variable_gen = task_variable_gen
        self.input_gen = input_gen
        self.sample_gen = sample_gen

        assert input_gen.required_task_variable_keys.issubset(task_variable_gen.task_variable_keys)
        assert sample_gen.required_task_variable_keys.issubset(task_variable_gen.task_variable_keys)
    
    def generate_trial_information(self, num_samples: int, **task_variables_kwargs) -> TrialInformation:
        task_variable_information = self.task_variable_gen.generate_variable_dict(**task_variables_kwargs)
        network_inputs = self.input_gen.generate_network_inputs(task_variable_information)
        sample_information = self.sample_gen.generate_sample_set(num_samples, task_variable_information)
        return TrialInformation(task_variable_information, network_inputs, sample_information)

