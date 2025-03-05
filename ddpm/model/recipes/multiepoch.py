from ddpm.model.main.base import *
from ddpm.model.residual import *
from ddpm.model.input import *
from ddpm.model.main.multiepoch import (
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
)


def teacher_forced_delayed_index_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, device
    )  # ... cueing epoch for indexing
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        underlying_sensory_shape[0],
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "epsilon_hat"
    return ddpm_model, mse_key


def teacher_forced_delayed_probe_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # no indexing unlike above
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        underlying_sensory_shape[0],
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "epsilon_hat"
    return ddpm_model, mse_key
