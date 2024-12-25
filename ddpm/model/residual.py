from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import List


from ddpm.model.unet import UNet




class ResidualModel(nn.Module):
    """
    ϵ_θ (x_t, t, Z)
    Just a linear-relu cascade but with an extra set of dims for time and inputs at each layer
    """
    def __init__(self, state_space_size: int, recurrence_hidden_layers: List[int], input_size: int, time_embedding_size: int) -> None:
        super().__init__()
        
        self.state_space_size = state_space_size
        self.recurrence_hidden_layers = recurrence_hidden_layers
        self.input_size = input_size
        self.time_embedding_size = time_embedding_size

        all_layer_sizes = recurrence_hidden_layers + [state_space_size]
        recurrence_layers = [nn.Linear(state_space_size + time_embedding_size + input_size, all_layer_sizes[0]), nn.ReLU()]        # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            # recurrence_layers.extend([nn.Linear(all_layer_sizes[i], op_s), nn.ReLU()])
            recurrence_layers.extend([nn.Linear(all_layer_sizes[i] + time_embedding_size + input_size, op_s), nn.ReLU()])
        recurrence_layers = recurrence_layers[:-1]
        
        self.layers = nn.ModuleList(recurrence_layers)  # R^N -> R^N
        
    def concatenate_with_time_and_input(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [B, T, layer_size]
        input_vector of shape [B, T, input_size]
        t_embeddings_schedule of shape [T, t_emb_size]
        """
        reshaped_t_schedule = t_embeddings_schedule.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x_concat = torch.concat([x, reshaped_t_schedule.to(x.device, x.dtype), input_vector.to(x.device, x.dtype)], -1)
        return x_concat

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [...], passed to all

        x[...,i,:] is x_{t_embeddings_schedule[i]}
            e.g. 
                if t_embeddings_schedule[i] = 1, then x[...,i,:] is x_1
                if t_embeddings_schedule[i] = 5, then x[...,i,:] is x_5
            But actually, the numerical values of t_embeddings_schedule won't be the integer values printed here of course
        """
        # x_concat = self.concatenate_with_time(x, t_schedule)
        # return self.model(x_concat) # sequential
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x_concat = self.concatenate_with_time_and_input(x, t_embeddings_schedule, input_vector)
                x = layer(x_concat)
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Softplus):
                x = layer(x)
            else:
                raise NotImplementedError
        return x
    


class UNetResidualModel(nn.Module):

    def __init__(self, image_size: int, input_size: int, time_embedding_size: int, num_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.input_size = input_size
        self.time_embedding_size = time_embedding_size
        self.num_channels = num_channels
        self.base_channels = base_channels

        total_input_vector_size = input_size + time_embedding_size
        self.unet = UNet(
            image_size = image_size, num_channels = num_channels, vector_dim = total_input_vector_size, base_channels = base_channels
        )

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [B, T, image_size, image_size]
        input_vector of shape [B, T, input_size]
        t_embeddings_schedule of shape [T, t_emb_size]
        """
        reshaped_t_schedule = t_embeddings_schedule.unsqueeze(0).repeat(x.shape[0], 1, 1)
        total_input_vector = torch.concat([input_vector, reshaped_t_schedule])
        return self.unet(x, total_input_vector)

