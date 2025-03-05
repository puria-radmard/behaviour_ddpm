from ddpm.tasks.variable.probabilistic import *
from ddpm.tasks.input.base import *
from ddpm.tasks.sample.parallel import *
from ddpm.tasks.main.base import *




def vectoral_even_causal_inference_parallel(
    sample_size, min_margin_div_pi: float, num_parallel_samples: int, **sample_kwargs
):
    task_variable_gen = StandardCausalInferenceTaskVariableGenerator(min_margin_div_pi)
    sensory_gen = JustReportSensoryGenerator(num_items=2)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size,
        **sample_kwargs,
        response_location_key="response_locations_cart"
    )
    sample_gen = ParallelExampleSampleGenerator(underlying_sample_generator=sample_gen, num_parallel_samples=num_parallel_samples)
    return DiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


