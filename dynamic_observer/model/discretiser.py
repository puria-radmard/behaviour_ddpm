import torch
from torch import Tensor as _T

from typing import Optional

from abc import ABC, abstractmethod


class DynamicsDiscretiser(ABC):

    @abstractmethod
    def step(self, x_k: _T, drift: _T, diffusion: _T, dt: _T, noise: Optional[_T] = None) -> _T:
        raise NotImplementedError
    

class EulerDiscretiser(DynamicsDiscretiser):

    def step(self, x_k: _T, drift: _T, diffusion: _T, dt: _T, noise: Optional[_T] = None) -> _T:
        if noise is None:
            noise = torch.randn_like(x_k)
        return x_k + (drift * dt) + (diffusion * dt.sqrt() * noise)

