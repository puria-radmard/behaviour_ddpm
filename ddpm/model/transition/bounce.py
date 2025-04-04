import torch
from torch import nn
from torch import Tensor as _T

from typing import Mapping, Any


from ddpm.model.transition.vectoral import VectoralResidualModel



class BouncePopulationResidualModel(VectoralResidualModel):
    """
    State is now shaped [2, D]
    Each state takes in the other one to get its own residual
    
    For now assume recurrence layers are the same, and all is 'biological'
    """
    def __init__(self, state_space_size: int, input_size: int, time_embedding_size: int) -> None:
        super().__init__(state_space_size, [], input_size, time_embedding_size, nonlin_first = True)
        self.bounceback_layers = nn.ModuleList(self.generate_layers([], include_time_and_input=False))

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        primary_to_bounce = super().forward(x[...,0,:], t_embeddings_schedule, input_vector)
        
        # bounce_to_primary = super().forward(x[...,1,:], t_embeddings_schedule, input_vector, override_layers = self.bounceback_layers)
        if self.nonlin_first:
            bounce_to_primary = torch.nn.functional.softplus(x[...,1,:])
        else:
            raise AttributeError
        for layer in self.bounceback_layers:
            bounce_to_primary = layer(bounce_to_primary)

        return torch.stack([bounce_to_primary, primary_to_bounce], dim=-2)    # reversed! [..., T, 2, D]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            print('LOADING VectoralResidualModel WEIGHTS INTO BouncePopulationResidualModel')
            assert set(state_dict.keys()) == {"layers.2.weight", "layers.2.bias", "layers.0.weight", "layers.0.bias"}
            state_dict['bounceback_layers.0.weight'] = state_dict.pop('layers.2.weight')[:,:self.state_space_size]
            state_dict['bounceback_layers.0.bias'] = state_dict.pop('layers.2.bias')
            assert strict and not assign
            return super(BouncePopulationResidualModel, self).load_state_dict(state_dict, strict, assign)
    


