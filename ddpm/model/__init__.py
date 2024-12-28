from ddpm.model.main import *
from ddpm.model.time_repr import *
from ddpm.model.residual import *
from ddpm.model.input import *

def standard_vectoral(sensory_shape, sample_shape, recurrence_hidden_layers, time_embedding_size, sigma2x_schedule, device, **model_kwargs):
    assert len(sensory_shape) == len(sample_shape) == 1
    input_model = InputModelBlock(sensory_shape, sensory_shape[0], device = device)
    residual_model = VectoralResidualModel(sample_shape[0], recurrence_hidden_layers, sensory_shape[0], time_embedding_size, **model_kwargs)
    ddpm_model = DDPMReverseProcessBase(sample_shape, sigma2x_schedule, residual_model, input_model, time_embedding_size, device)
    return ddpm_model


def standard_vectoral_in_images_out(sensory_shape, sample_shape, base_unet_channels, time_embedding_size, sigma2x_schedule, device):
    (num_input_channels, image_size, image_size_alt) = sample_shape
    assert image_size == image_size_alt
    input_model = InputModelBlock(sensory_shape, sensory_shape[0], device = device)
    residual_model = UNetResidualModel(image_size, sensory_shape[0], time_embedding_size, num_input_channels, base_unet_channels)
    ddpm_model = DDPMReverseProcessBase(sample_shape, sigma2x_schedule, residual_model, input_model, time_embedding_size, device)
    return ddpm_model

