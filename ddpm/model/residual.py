from __future__ import annotations

import torch
from torch import nn
from torch import vmap
from torch import Tensor as _T

from typing import List, Mapping, Any


from ddpm.model.unet import UNet


class VectoralResidualModel(nn.Module):
    """
    ϵ_θ (x_t, t, Z)
    """

    def __init__(
        self,
        state_space_size: int,
        recurrence_hidden_layers: List[int],
        input_size: int,
        time_embedding_size: int,
        nonlin_first: bool,
    ) -> None:

        super().__init__()

        self.input_size = input_size
        self.state_space_size = state_space_size
        self.recurrence_hidden_layers = recurrence_hidden_layers
        self.time_embedding_size = time_embedding_size
        self.nonlin_first = nonlin_first

        all_layer_sizes = recurrence_hidden_layers + [state_space_size]
        recurrence_layers = []
        recurrence_layers.extend(
            [
                nn.Linear(
                    state_space_size + time_embedding_size + input_size,
                    all_layer_sizes[0],
                ),
                nn.Softplus(),
            ]
        )  # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            recurrence_layers.extend(
                [nn.Linear(all_layer_sizes[i], op_s), nn.Softplus()]
            )
        recurrence_layers = recurrence_layers[:-1]

        self.layers = nn.ModuleList(recurrence_layers)  # R^N -> R^N

    @staticmethod
    def unsqueeze_start_dims(tensor: _T, start_dims: List[int]):
        return tensor[*[None for _ in start_dims]].expand(
            *start_dims, *tensor.shape
        )

    def concatenate_with_time_and_input(
        self, x: _T, t_embeddings_schedule: _T, input_vector: _T
    ) -> _T:
        """
        x of shape [..., T, layer_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [..., T, input_size]
        """
        reshaped_t_schedule = self.unsqueeze_start_dims(
            t_embeddings_schedule, x.shape[:-2]
        )
        x_concat = torch.concat(
            [
                x,
                reshaped_t_schedule.to(x.device, x.dtype),
                input_vector.to(x.device, x.dtype),
            ],
            -1,
        )
        return x_concat

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [..., T, input_size],             <s> passed to all </s>

        x[...,i,:] is x_{t_embeddings_schedule[i]}
            e.g.
                if t_embeddings_schedule[i] = 1, then x[...,i,:] is x_1
                if t_embeddings_schedule[i] = 5, then x[...,i,:] is x_5
            But actually, the numerical values of t_embeddings_schedule won't be the integer values printed here of course
        """
        if self.nonlin_first:
            x = torch.nn.functional.softplus(x)
        x = self.concatenate_with_time_and_input(x, t_embeddings_schedule, input_vector)
        for layer in self.layers:
            x = layer(x)
        return x


class UNetResidualModel(nn.Module):

    def __init__(
        self,
        image_size: int,
        input_size: int,
        time_embedding_size: int,
        num_channels: int,
        base_channels: int = 64,
    ) -> None:

        self.input_size = input_size

        super().__init__()
        self.image_size = image_size
        self.time_embedding_size = time_embedding_size
        self.num_channels = num_channels
        self.base_channels = base_channels

        print("NOT PASSING INPUT TO UNET FOR NOW!!!")

        total_input_vector_size = time_embedding_size  #  + input_size
        self.unet = UNet(
            image_size=image_size,
            num_channels=num_channels,
            vector_dim=total_input_vector_size,
            base_channels=base_channels,
        )

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [B, T, num_channels, image_size, image_size]
        t_embeddings_schedule of shape [T, t_emb_size]
        input_vector of shape [B, T, input_size]
        """
        raise Exception("Make start dims general again!")
        reshaped_t_schedule = t_embeddings_schedule.unsqueeze(0).expand(
            x.shape[0], 1, 1
        )
        total_input_vector = reshaped_t_schedule  # torch.concat([input_vector, reshaped_t_schedule], -1).float()
        return self.unet(x, total_input_vector)
