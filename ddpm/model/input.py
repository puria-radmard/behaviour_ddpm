from __future__ import annotations

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
        assert x.shape[-1] == self.network_input_size
        extra_dims = len(x.shape) - 1
        return x[...,None,:].repeat(*[1]*extra_dims, num_timesteps, 1)

    def forward(self, x: _T, num_timesteps: int) -> _T:
        assert (
            tuple(x.shape[-len(self.sensory_shape) :]) == self.sensory_shape
        ), f"Expected inputs ending with shape {self.sensory_shape}, got shape {x.shape}"
        return self.add_timesteps(x, num_timesteps).to(self.device)


class AllowIndexInputModelBlock(InputModelBlock):
    """
    Meant to work with e.g. DelayedIndexCuingSensoryGeneratorWithMemory
    """

    def __init__(
        self, underlying_input_block: InputModelBlock, num_indices: int, device: str
    ) -> None:
        super(InputModelBlock, self).__init__()
        self.device = device
        self.underlying_input_block = underlying_input_block.to(device)
        self.sensory_shape = self.underlying_input_block.sensory_shape
        self.network_input_size = self.underlying_input_block.network_input_size
        self.index_embeddings = nn.Embedding(
            num_indices, underlying_input_block.network_input_size
        )

    def forward(self, x: _T, num_timesteps: int) -> _T:
        if tuple(x.shape[-len(self.sensory_shape) :]) == self.sensory_shape:
            ret = self.underlying_input_block(x, num_timesteps)
        else:
            assert (
                x.shape[-1] == 1
            ), f"AllowIndexInputModelBlock expects inputs with shape ending in either {self.sensory_shape} or {[1]}, not {x.shape}"
            ret = self.index_embeddings(x.to(self.device))[..., 0, :]
            ret = self.add_timesteps(ret, num_timesteps)
        return ret
