from torch import Tensor as _T

from typing import Dict, List, Optional

import numpy as np

from dataclasses import dataclass

from ddpm.tasks.sample.base import ExampleSampleGenerator, SwapSampleInformation
from ddpm.tasks.main.base import DiffusionTask
from ddpm.tasks.variable.base import TaskVariableGenerator
from ddpm.tasks.variable.preinitialised import InitialisedSampleSpaceTaskVariableGenerator
from ddpm.tasks.input.multiepoch import MultiEpochSensoryGenerator


@dataclass
class MultiepochTrialInformation:
    task_variable_information: Dict[str, _T]
    prep_network_inputs: List[_T]
    diffusion_network_inputs: List[_T]
    prep_epoch_durations: List[int]
    diffusion_epoch_durations: List[Optional[int]]
    sample_information: SwapSampleInformation
    pre_prep_sample_information: Optional[SwapSampleInformation] = None



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
        self,
        batch_size: Optional[int], 
        num_samples: int, 
        *_,
        override_task_variable_information: Optional[Dict[str, _T]] = None,
        **task_variables_kwargs
    ) -> MultiepochTrialInformation:
        if override_task_variable_information is None:
            task_variable_information = self.task_variable_gen.generate_variable_dict(
                batch_size=batch_size, **task_variables_kwargs
            )
        else:
            task_variable_information = override_task_variable_information
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
        sample_information = self.sample_gen.generate_sample_set(
            num_samples, task_variable_information
        )
        prep_epoch_durations = task_variable_information["prep_epoch_durations"]
        diffusion_epoch_durations = task_variable_information["diffusion_epoch_durations"]
        return MultiepochTrialInformation(
            task_variable_information,
            prep_network_inputs,
            diffusion_network_inputs,
            prep_epoch_durations,
            diffusion_epoch_durations,
            sample_information,
        )

    def generate_test_trial_information(
        self, num_samples: int, **task_variables_kwargs
    ):
        """
        For tasks with multiple trial types (e.g. different feature dimensions can be cued, different ordering of cue and stimulus),
        generate one trial type for each possible combination
        """
        representative_task_information = self.task_variable_gen.generate_representative_variable_dict(
            **task_variables_kwargs
        )
        return self.generate_trial_information(
            batch_size=None,
            num_samples=num_samples,
            override_task_variable_information=representative_task_information
        )





class InitialisedSampleSpaceMultiEpochDiffusionTask(MultiEpochDiffusionTask):
    """
    Same as above except we also provide some sample space activity in the pre_prep_sample_information field
    """

    def __init__(self, task_variable_gen: InitialisedSampleSpaceTaskVariableGenerator, sensory_gen: MultiEpochSensoryGenerator, sample_gen: ExampleSampleGenerator) -> None:
        super().__init__(task_variable_gen, sensory_gen, sample_gen)
        self.task_variable_gen: InitialisedSampleSpaceTaskVariableGenerator # For typing later

        assert sensory_gen.required_task_variable_keys.issubset(
            task_variable_gen.pre_prep_variable_keys
        )

    def generate_trial_information(self, batch_size: int, num_samples: int, **task_variables_kwargs) -> MultiepochTrialInformation:
        ret = super().generate_trial_information(batch_size, num_samples, **task_variables_kwargs)

        import pdb; pdb.set_trace(header = 'Generate pre_prep_sample_information using relevant keys from task_variable_information')

        pre_prep_sample_information = self.sample_gen.generate_sample_set(
            num_samples, ret.task_variable_information['pre_prep_info']
        )

        ret.pre_prep_sample_information = pre_prep_sample_information

        return ret

