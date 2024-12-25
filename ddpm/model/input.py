from __future__ import annotations

from torch import nn
from torch import Tensor as _T

from typing import Tuple



class InputModelBlock(nn.Module):
    """
    Takes in structured input and outputs the same thing
    """

    def __init__(self, input_shape: Tuple[int], output_shape: int, *args, **kwargs) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert self.output_shape == self.input_shape
        super().__init__(*args, **kwargs)
    
    def forward(self, x: _T) -> _T:
        assert tuple(x.shape[-len(self.input_shape):]) == self.input_shape, f"Expected inputs ending with shape {self.input_shape}, got shape {x.shape}"
        return x

