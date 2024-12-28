from ddpm.tasks.task_variable import *
from ddpm.tasks.input import *
from ddpm.tasks.sample import *
from ddpm.tasks.main import *


def standard_fixed_probability_vectoral(sample_size, pmf: List[int], **sample_kwargs):
    n_items = len(pmf)
    task_variable_gen = FixedProvidedSwapProbabilityTaskVariableGenerator(n_items, pmf)
    sensory_gen = ProvidedSwapProbabilitySensoryGenerator(n_items)
    sample_gen = VectoralExampleSampleGenerator(sample_size = sample_size, **sample_kwargs)
    return WMDiffusionTask(task_variable_gen=task_variable_gen, sensory_gen=sensory_gen, sample_gen=sample_gen)


def indexing_cue_vectoral(sample_size, n_items, **sample_kwargs):
    task_variable_gen = ZeroTemperatureSwapProbabilityTaskVariableGenerator(n_items)
    sensory_gen = IndexCuingSensoryGenerator(n_items)
    sample_gen = VectoralExampleSampleGenerator(sample_size = sample_size, **sample_kwargs)
    return WMDiffusionTask(task_variable_gen=task_variable_gen, sensory_gen=sensory_gen, sample_gen=sample_gen)


def probe_cue_vectoral(sample_size, n_items, **sample_kwargs):
    task_variable_gen = ZeroTemperatureSwapProbabilityTaskVariableGenerator(n_items)
    sensory_gen = ProbeCuingSensoryGenerator(n_items)
    sample_gen = VectoralExampleSampleGenerator(sample_size = sample_size, **sample_kwargs)
    return WMDiffusionTask(task_variable_gen=task_variable_gen, sensory_gen=sensory_gen, sample_gen=sample_gen)


def standard_fixed_probability_vectoral_strip_image_in(image_size: int, strip_pixel_width: int, pmf: List[int]):
    n_items = len(pmf)
    task_variable_gen = FixedProvidedSwapProbabilityTaskVariableGenerator(n_items, pmf)
    sensory_gen = ProvidedSwapProbabilitySensoryGenerator(n_items)
    sample_gen = WhiteNoiseStripExampleSampleGenerator(image_size = image_size, strip_pixel_width = strip_pixel_width)
    return WMDiffusionTask(task_variable_gen=task_variable_gen, sensory_gen=sensory_gen, sample_gen=sample_gen)
