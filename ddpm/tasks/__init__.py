from ddpm.tasks.task_variable import *
from ddpm.tasks.task_variable_probabilistic import *
from ddpm.tasks.input import *
from ddpm.tasks.sample import *
from ddpm.tasks.main import *
from ddpm.tasks.multiepoch_tasks import *
from ddpm.tasks.multiepoch_input import *


def standard_fixed_probability_vectoral(sample_size, pmf: List[int], **sample_kwargs):
    n_items = len(pmf)
    task_variable_gen = FixedProvidedSwapProbabilityTaskVariableGenerator(n_items, pmf)
    sensory_gen = ProvidedSwapProbabilitySensoryGenerator(n_items)
    polar_sample = sample_kwargs.pop("polar_sample")
    if polar_sample:
        sample_gen = RadialVectoralEmbeddedExampleSampleGenerator(
            sample_size=sample_size, **sample_kwargs
        )
    else:
        sample_gen = VectoralEmbeddedExampleSampleGenerator(
            sample_size=sample_size, **sample_kwargs
        )
    return DiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def vectoral_even_causal_inference(
    sample_size, min_margin_div_pi: float, **sample_kwargs
):
    task_variable_gen = StandardCausalInferenceTaskVariableGenerator(min_margin_div_pi)
    sensory_gen = JustReportSensoryGenerator(num_items=2)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size,
        **sample_kwargs,
        response_location_key="response_locations_cart"
    )
    return DiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def anchored_vectoral_bayesian_multisensory_causal_inference(
    sample_size: int,
    min_margin_div_pi: float,
    
):
    pass


def indexing_cue_vectoral(sample_size, n_items, **sample_kwargs):
    task_variable_gen = ZeroTemperatureSwapProbabilityTaskVariableGenerator(n_items)
    sensory_gen = IndexCuingSensoryGenerator(n_items)
    polar_sample = sample_kwargs.pop("polar_sample")
    if polar_sample:
        sample_gen = RadialVectoralEmbeddedExampleSampleGenerator(
            sample_size=sample_size, **sample_kwargs
        )
    else:
        sample_gen = VectoralEmbeddedExampleSampleGenerator(
            sample_size=sample_size, **sample_kwargs
        )
    return DiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def probe_cue_vectoral(sample_size, n_items, **sample_kwargs):
    task_variable_gen = ZeroTemperatureSwapProbabilityTaskVariableGenerator(n_items)
    sensory_gen = ProbeCuingSensoryGenerator(n_items)
    polar_sample = sample_kwargs.pop("polar_sample")
    if polar_sample:
        sample_gen = RadialVectoralEmbeddedExampleSampleGenerator(
            sample_size=sample_size, **sample_kwargs
        )
    else:
        sample_gen = VectoralEmbeddedExampleSampleGenerator(
            sample_size=sample_size, **sample_kwargs
        )
    return DiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def delayed_indexing_cue_fixed_probability_vectoral(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    index_duration,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, index_duration
    )
    sensory_gen = DelayedIndexCuingSensoryGeneratorWithMemory(num_items=num_items)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def standard_fixed_probability_vectoral_strip_image_in(
    image_size: int, strip_pixel_width: int, pmf: List[float]
):
    n_items = len(pmf)
    task_variable_gen = FixedProvidedSwapProbabilityTaskVariableGenerator(n_items, pmf)
    sensory_gen = ProvidedSwapProbabilitySensoryGenerator(n_items)
    sample_gen = WhiteNoiseStripExampleSampleGenerator(
        image_size=image_size, strip_pixel_width=strip_pixel_width
    )
    return DiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )
