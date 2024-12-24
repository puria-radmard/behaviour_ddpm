from __future__ import annotations

from torch import nn
from torch import Tensor as _T

from typing import Tuple



class InputModelBlock(nn.Module):
    """
    Takes in structured input and outputs the same thing
    """

    def __init__(self, input_shape: Tuple[int], output_size: int, *args, **kwargs) -> None:
        self.input_shape = input_shape
        self.output_size = output_size
        super().__init__(*args, **kwargs)
        assert input_shape[0] == output_size and len(input_shape) == 1  # For the basic case!
    
    def forward(self, x: _T) -> _T:
        assert tuple(x.shape[-len(self.input_shape):]) == self.input_shape, f"Expected inputs ending with shape {self.input_shape}, got shape {x.shape}"
        return x

