import torch
from torch import nn
from torch import Tensor as _T

from typing import List


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

        recurrence_layers = self.generate_layers(recurrence_hidden_layers)
        self.layers = nn.ModuleList(recurrence_layers)  # R^N -> R^N

    def generate_layers(self, recurrence_hidden_layers, include_time_and_input = True) -> List[nn.Module]:
        all_layer_sizes = recurrence_hidden_layers + [self.state_space_size]
        recurrence_layers = []
        total_input_size = self.state_space_size + self.time_embedding_size + self.input_size if include_time_and_input else self.state_space_size
        recurrence_layers.extend([nn.Linear(total_input_size,all_layer_sizes[0]), nn.Softplus()])  # 1 to include time also!
        for i, op_s in enumerate(all_layer_sizes[1:]):
            recurrence_layers.extend(
                [nn.Linear(all_layer_sizes[i], op_s), nn.Softplus()]
            )
        recurrence_layers = recurrence_layers[:-1]

        return recurrence_layers


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

