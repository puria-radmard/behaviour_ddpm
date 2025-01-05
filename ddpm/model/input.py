from __future__ import annotations

from torch import nn
from torch import Tensor as _T

from typing import Tuple



class InputModelBlock(nn.Module):
    """
    Sensory input (structured) --> input to residual network (always vectoral)
    """

    def __init__(self, sensory_shape: Tuple[int], network_input_size: int, device: str) -> None:
        self.sensory_shape = sensory_shape
        self.network_input_size = network_input_size
        self.device = device
        assert len(sensory_shape) == 1 and sensory_shape[0] == network_input_size,\
            f"InputModelBlock can only currently handle the identity mapping. Cannot have sensory_shape = {sensory_shape} and network_input_size = {network_input_size}"
        super().__init__()
    
    def forward(self, x: _T) -> _T:
        assert tuple(x.shape[-len(self.sensory_shape):]) == self.sensory_shape, f"Expected inputs ending with shape {self.sensory_shape}, got shape {x.shape}"
        return x.to(self.device)

