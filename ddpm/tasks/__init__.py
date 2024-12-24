from sampling_ddpm.ddpm.tasks.task_variable import *
from sampling_ddpm.ddpm.tasks.input import *
from sampling_ddpm.ddpm.tasks.sample import *
from sampling_ddpm.ddpm.tasks.main import *


def standard_fixed_probability_tabular(pmf: List[int]):
    n_items = len(pmf)
    task_variable_gen = FixedProvidedSwapProbabilityTaskVariableGenerator(n_items, pmf)
    input_gen = ProvidedSwapProbabilityInputGenerator(n_items)
    sample_gen = TabularExampleSampleGenerator()
    return WMDiffusionTask(task_variable_gen=task_variable_gen, input_gen=input_gen, sample_gen=sample_gen)

