from ddpm.model.main import *
from ddpm.model.time_repr import *
from ddpm.model.residual import *
from ddpm.model.input import *

def standard_tabular(input_shape, state_space_size, recurrence_hidden_layers, time_embedding_size, sigma2x_schedule):
    input_model = InputModelBlock(input_shape, input_shape)
    residual_model = ResidualModel(state_space_size, recurrence_hidden_layers, input_shape[0], time_embedding_size)
    ddpm_model = EmbeddedTabularDDPMReverseProcess(state_space_size, residual_model, input_model, sigma2x_schedule, time_embedding_size)
    return ddpm_model


def standard_tabular_in_images_out(input_shape, image_size, num_input_channels, base_unet_channels, time_embedding_size, sigma2x_schedule):
    input_model = InputModelBlock(input_shape, input_shape)
    residual_model = UNetResidualModel(image_size, input_shape[0], time_embedding_size, num_input_channels, base_channels)
    ddpm_model = EmbeddedTabularDDPMReverseProcess(state_space_size, residual_model, input_model, sigma2x_schedule, time_embedding_size)
    return ddpm_model




