from ddpm.tasks.variable.base import *
from ddpm.tasks.main.multiepoch_scorematching import *
from ddpm.tasks.input.multiepoch import *
from ddpm.tasks.distribution import *
from ddpm.tasks.variable.probabilistic import *
from ddpm.tasks.input.base import *
from ddpm.tasks.main.base import *
from ddpm.tasks.main.scorematching import *



def score_matching_delayed_probe_cue_vectoral_with_swap_function(
    num_items,
    swap_function_width,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    device = 'cuda',
    **distribution_kwargs,
):
    task_variable_gen = ProbeDistanceProbabilityTaskVariableGenerator(
        num_items, swap_function_width, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemory(num_items=num_items)
    distribution_gen = DiagonalGaussianOnCircleMixtureDistributionInformationGenerator(
        **distribution_kwargs, device = device
    )
    return ScoreMatchingMultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        distribution_gen=distribution_gen,
    )



def score_matching_vectoral_even_causal_inference(
    min_margin_div_pi: float, device = 'cuda', **distribution_kwargs
):
    task_variable_gen = StandardCausalInferenceTaskVariableGenerator(min_margin_div_pi)
    sensory_gen = JustReportSensoryGenerator(num_items=2)
    distribution_gen = DiagonalGaussianOnCircleMixtureDistributionInformationGenerator(
        **distribution_kwargs, 
        response_location_key="response_locations_cart",
        device = device
    )
    return ScoreMatchingDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        distribution_gen=distribution_gen,
        
    )

    

