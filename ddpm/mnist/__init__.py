import os
from typing import List

from torch import Tensor as _T

from ddpm.tasks.main.multiepoch import MultiEpochDiffusionTask
from ddpm.mnist.task import (
    MNISTClassExampleSampleGenerator,
    TimestepCounterTaskVariableGenerator,
    TimestepCounterSensoryGenerator,
    ClassificationTaskVariableGenerator,
    ClassificationPlusMinusTaskVariableGenerator,
    ClassificationSensoryGenerator,
)

from ddpm.model.main.base import *
from ddpm.model.transition import VectoralResidualModel
from ddpm.model.input import InputModelBlock, AllowIndexInputModelBlock
from ddpm.model.main.multiepoch import (
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
)


def mnist_classification_task(vae_base_path: str, z_dim, **task_variable_gen_kwargs):
    task_variable_gen = ClassificationTaskVariableGenerator(
        num_classes=10, **task_variable_gen_kwargs
    )
    sensory_gen = ClassificationSensoryGenerator(num_classes=10)
    sample_gen = MNISTClassExampleSampleGenerator(
        z_dim,
        os.path.join(vae_base_path, "vae_state.mdl"),
        os.path.join(vae_base_path, "vae_latent_class_moments.data"),
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def mnist_classification_plus_minus_task(
    vae_base_path: str, z_dim, **task_variable_gen_kwargs
):
    task_variable_gen = ClassificationPlusMinusTaskVariableGenerator(
        num_classes=10, **task_variable_gen_kwargs
    )
    sensory_gen = ClassificationSensoryGenerator(num_classes=10)
    sample_gen = MNISTClassExampleSampleGenerator(
        z_dim,
        os.path.join(vae_base_path, "vae_state.mdl"),
        os.path.join(vae_base_path, "vae_latent_class_moments.data"),
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def mnist_counting_task(vae_base_path: str, z_dim, **task_variable_gen_kwargs):
    task_variable_gen = TimestepCounterTaskVariableGenerator(
        num_classes=10, **task_variable_gen_kwargs
    )
    sensory_gen = TimestepCounterSensoryGenerator()
    sample_gen = MNISTClassExampleSampleGenerator(
        z_dim,
        os.path.join(vae_base_path, "vae_state.mdl"),
        os.path.join(vae_base_path, "vae_latent_class_moments.data"),
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def mnist_ddpm_model_embedding_input_general(
    num_inputs: int,
    z_dim: int,
    sample_ambient_dim: int,
    recurrence_hidden_layers: List[int],
    input_shape: int,
    time_embedding_size: int,
    sigma2x_schedule: _T,
    device: str,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    under_input_model = InputModelBlock(
        (input_shape,), input_shape, device=device
    )  # Doesn't matter at all, just passed onto next line
    input_model = AllowIndexInputModelBlock(under_input_model, num_inputs, device)
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_shape,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=[
            z_dim,
        ],
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    return ddpm_model


def mnist_classification_model(**kwargs):
    return mnist_ddpm_model_embedding_input_general(
        num_inputs=11, **kwargs
    )  # 10 classes and one diffusion token


def mnist_counting_model(**kwargs):
    return mnist_ddpm_model_embedding_input_general(
        num_inputs=4, **kwargs
    )  # 4 different unique phases
