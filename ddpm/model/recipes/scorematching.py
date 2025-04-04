from ddpm.model.main.base import *
from ddpm.model.transition import *
from ddpm.model.input import *
#from ddpm.model.main.scorematching import (
#    ScoreMatchingMultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
#    ScoreMatchingLinearSubspaceTeacherForcedDDPMReverseProcess
#)



def score_matching_teacher_forced_delayed_probe_cue(
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
    ddpm_model = ScoreMatchingMultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "score_hat"
    return ddpm_model, mse_key




def score_matching_teacher_forced_vectoral(
    sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert len(sensory_shape) == len(sample_shape) == 1
    input_model = InputModelBlock(sensory_shape, sensory_shape[0], device=device)
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        sensory_shape[0],
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = ScoreMatchingLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "score_hat"
    return ddpm_model, mse_key


