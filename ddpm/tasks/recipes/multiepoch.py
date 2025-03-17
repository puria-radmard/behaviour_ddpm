from ddpm.tasks.variable.base import *
from ddpm.tasks.sample.base import *
from ddpm.tasks.main.multiepoch import *
from ddpm.tasks.input.multiepoch import *





def delayed_indexing_cue_fixed_probability_vectoral(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, 
        pre_index_delay_duration, index_duration, post_index_delay_duration
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




def delayed_probe_cue_vectoral_spike_and_slab(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemory(num_items=num_items)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )



def delayed_probe_cue_vectoral_spike_and_slab_palimpsest(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    probe_num_tc,
    report_num_tc,
    probe_num_width,
    report_num_width,
    vectorise_input: bool = True,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest(
        num_items=num_items,
        probe_num_tc=probe_num_tc,
        report_num_tc=report_num_tc,
        probe_num_width=probe_num_width,
        report_num_width=report_num_width,
        vectorise_input=vectorise_input,
    )
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )




def delayed_probe_cue_vectoral_with_swap_function(
    sample_size,
    num_items,
    swap_function_width,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    **sample_kwargs
):
    task_variable_gen = ProbeDistanceProbabilityTaskVariableGenerator(
        num_items, swap_function_width, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemory(num_items=num_items)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def delayed_probe_cue_vectoral_with_swap_function_palimpsest(
    sample_size,
    num_items,
    swap_function_width,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    probe_num_tc,
    report_num_tc,
    probe_num_width,
    report_num_width,
    vectorise_input: bool = True,
    **sample_kwargs
):
    task_variable_gen = ProbeDistanceProbabilityTaskVariableGenerator(
        num_items, swap_function_width, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest(
        num_items=num_items,
        probe_num_tc=probe_num_tc,
        report_num_tc=report_num_tc,
        probe_num_width=probe_num_width,
        report_num_width=report_num_width,
        vectorise_input=vectorise_input,
    )
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )



def sequentially_presented_vectoral_even_causal_inference(
    stimulus_exposure_durations, delay_durations,
    sample_size, min_margin_div_pi: float, **sample_kwargs
):
    task_variable_gen = SequentialCausalInferenceTaskVariableGenerator(min_margin_div_pi)
    sensory_gen = SequentialJustReportSensoryGenerator(num_items=2)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size,
        **sample_kwargs,
        response_location_key="response_locations_cart"
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )
