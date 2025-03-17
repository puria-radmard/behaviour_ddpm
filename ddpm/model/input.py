from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Tuple


class InputModelBlock(nn.Module):
    """
    Sensory input (structured) --> input to residual network (always vectoral)
    """

    def __init__(
        self, sensory_shape: Tuple[int], network_input_size: int, device: str
    ) -> None:
        self.sensory_shape = tuple(sensory_shape)
        self.network_input_size = network_input_size
        self.device = device
        assert (
            len(sensory_shape) == 1 and sensory_shape[0] == network_input_size
        ), f"InputModelBlock can only currently handle the identity mapping. Cannot have sensory_shape = {sensory_shape} and network_input_size = {network_input_size}"
        super().__init__()

    def add_timesteps(self, x: _T, num_timesteps: int) -> _T:
        assert x.shape[-1] == self.network_input_size, x.shape
        extra_dims = len(x.shape) - 1
        target_shape = list(x.shape)
        target_shape.insert(extra_dims, num_timesteps)
        return x[...,None,:].expand(*target_shape)

    def forward(self, x: _T, num_timesteps: int) -> _T:
        assert (
            tuple(x.shape[-len(self.sensory_shape) :]) == tuple(self.sensory_shape)
        ), f"Expected inputs ending with shape {self.sensory_shape}, got shape {x.shape}"
        return self.add_timesteps(x, num_timesteps).to(self.device)



class FactorisedInputModelBlock(InputModelBlock):

    def __init__(self, sensory_shape: Tuple[int], feature_projection_sizes: Tuple[int], device: str) -> None:
        super(InputModelBlock, self).__init__()
        self.device = device
        self.sensory_shape = sensory_shape
        self.feature_projection_sizes = feature_projection_sizes
        self.network_input_size = feature_projection_sizes[0] * feature_projection_sizes[1]
        assert len(sensory_shape) == len(feature_projection_sizes) == 2

        self.feature_0_proj = nn.Linear(sensory_shape[0], feature_projection_sizes[0])
        self.feature_1_proj = nn.Linear(sensory_shape[1], feature_projection_sizes[1])

    def forward(self, x: _T, num_timesteps: int) -> _T:
        x_feature_1_proj = self.feature_1_proj(x.to(self.device).float())    # [..., D1, N1]
        x_feature_0_proj = self.feature_0_proj(x_feature_1_proj.transpose(-2, -1))  # [..., N1, N2]
        x_flattened = x_feature_0_proj.reshape(*x_feature_0_proj.shape[:-2], -1)    # [..., N1*N2]
        x_expand = self.add_timesteps(x_flattened, num_timesteps).to(self.device)
        return x_expand
        


class AllowIndexInputModelBlock(InputModelBlock):
    """
    Meant to work with e.g. DelayedIndexCuingSensoryGeneratorWithMemory
    """

    def __init__(
        self, underlying_input_block: InputModelBlock, num_indices: int, indexing_embeddings_same_slots: bool, device: str
    ) -> None:
        super(InputModelBlock, self).__init__()
        self.device = device
        self.underlying_input_block = underlying_input_block.to(device)
        self.sensory_shape = self.underlying_input_block.sensory_shape
        self.num_indices = num_indices
        self.indexing_embeddings_same_slots = indexing_embeddings_same_slots
        if indexing_embeddings_same_slots:
            self.network_input_size = self.underlying_input_block.network_input_size
            self.index_embeddings = nn.Embedding(
                num_indices, underlying_input_block.network_input_size
            )
        else:
            self.network_input_size = self.underlying_input_block.network_input_size + self.num_indices
            self.index_embeddings = nn.Embedding(num_indices, num_indices)
        pass

    def forward(self, x: _T, num_timesteps: int) -> _T:
        if tuple(x.shape[-len(self.sensory_shape) :]) == self.sensory_shape:
            ret = self.underlying_input_block(x, num_timesteps) # [B, S, T, D]
            if not self.indexing_embeddings_same_slots:
                empty_index = torch.zeros(*ret.shape[:-1], self.num_indices, device = ret.device, dtype = ret.dtype)
                ret = torch.concat([ret, empty_index], dim = -1)
        else:
            assert (
                x.shape[-1] == 1
            ), f"AllowIndexInputModelBlock expects inputs with shape ending in either {self.sensory_shape} or 1, not {x.shape}"
            if self.indexing_embeddings_same_slots:
                ret = self.index_embeddings(x.to(self.device))[..., 0, :]
            else:
                embs = self.index_embeddings(x.to(self.device))[..., 0, :]
                empty_sensory = torch.zeros(*embs.shape[:-1], self.underlying_input_block.network_input_size, device = embs.device, dtype = embs.dtype)
                ret = torch.concat([empty_sensory, embs], dim = -1)
            ret = self.add_timesteps(ret, num_timesteps)
        return ret
