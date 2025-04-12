import torch
from torch import nn
from torch import Tensor as _T

from typing import List

from math import prod


class DendriticIntegrationNode(nn.Module):

    """
    Represents mapping from activities at some fanned-out stage in the dendritic tree
        to the next most proximal layer.
    
    High level calc:
        Input shaped [N, B1, B2, ..., BL] where:
            N is the number of neurons
            Bi is the branching factor at the ith layer of the tree
            BL is the current layer's branching factor

        Output shaped [N, B1, B2, ..., B_{L-1}]
        
        Importantly, there are N * B1 * ... * BL elements here,
            and groups of BL of them are to be projected to N * B1 * ... * B_{L-1} elements
        
        Each projection has its own weights,
            i.e. take one path from neuron n (proximal -> distal) in the dendritic tree:
            
            inputs[n, b1, b2, ..., b_{L-1}, :] -> outputs[n, b1, b2, ..., b_{L-1}]
    """

    def __init__(
        self,
        num_neurons: int,
        branching_factors: List[int],
        time_representation_dim: int,
    ) -> None:
        super().__init__()

        self.num_neurons = num_neurons
        self.branching_factors = branching_factors
        self.time_representation_dim = time_representation_dim

        self.register_parameter('propagation_weights', torch.nn.Parameter(torch.zeros(num_neurons, *branching_factors)))
        nn.init.xavier_normal_(self.propagation_weights)

        self.time_representation_weights = nn.Linear(time_representation_dim, num_neurons * prod(branching_factors[:-1]), bias = True)
    
    def _check_activation_shape(self, activations: _T, t_embeddings_schedule: _T) -> int:
        expected_end_shape = tuple([self.num_neurons, *self.branching_factors])
        assert tuple(activations.shape[-1-len(self.branching_factors):]) == expected_end_shape, \
            f"Got activations of shape {activations.shape}, expected it to end with {expected_end_shape}"
        assert (activations.shape[-len(self.branching_factors)-2], self.time_representation_dim) == tuple(t_embeddings_schedule.shape)
        return len(activations.shape) - 1 - len(self.branching_factors)

    def preactivation(self, activations: _T, t_embeddings_schedule: _T) -> List[_T]:
        extra_dims = self._check_activation_shape(activations = activations, t_embeddings_schedule = t_embeddings_schedule)

        projected_t_embeddings_schedule = self.time_representation_weights(t_embeddings_schedule)
        reshaped_projected_t_embeddings_schedule = projected_t_embeddings_schedule.reshape(
            t_embeddings_schedule.shape[0], self.num_neurons, *self.branching_factors[:-1]
        )

        preact_dendrite = (self.propagation_weights[*[None]*extra_dims] * activations).sum(-1)

        return preact_dendrite, reshaped_projected_t_embeddings_schedule

    def forward(self, activations: _T, *other_inputs: _T) -> _T:
        """
        Expecting:
            activations shaped [..., T, num_nodes, <branching factors>]
            t_embeddings_schedule of shape [T, time_emb_size]
        Output of shape [..., num_nodes]
        """
        preact_axon, *preac_externals = self.preactivation(activations, *other_inputs)
        preact = preact_axon
        for preac_external in preac_externals:
            num_extra_dims = len(preact.shape) - len(preac_external.shape)
            preact = preact + preac_external[*[None]*num_extra_dims].expand(preact.shape)
        activation = torch.nn.functional.softplus(preact)
        return activation



class DendriticIntegrationFanoutNode(nn.Module):

    """
    High-level:
        maps from [num_neuron] to [num_neurons, <all branching_factors>]
        this is a fully connected layer, so the ordering of axes doesn't really matter
        all that matters is that in subsequent DendriticIntegrationNode trees,
            activations are kept private to each subpopulation of nodes... see docstring for that
    """

    def __init__(self, num_neurons: int, branching_factors: List[int], time_representation_dim: int, input_size: int) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.num_nodes = num_neurons * prod(branching_factors)
        self.branching_factors = branching_factors
        self.input_size = input_size
        self.time_representation_dim = time_representation_dim
        
        self.axon_weights = nn.Linear(num_neurons, self.num_nodes, bias = True)
        self.time_weights = nn.Linear(time_representation_dim, self.num_nodes, bias = False)
        self.input_weights = nn.Linear(input_size, self.num_nodes, bias = False)

    def forward(self, activations: _T, t_embeddings_schedule: _T, input_vector: _T):
        """
        Expecting:
            activations shaped [..., T, num_neurons]
            t_embeddings_schedule of shape [T, time_emb_size]
            input_vector of shape [..., T, input_size]
        
        Output of shape [..., num_neurons, <all branching_factors>]
            This output will get subsequentially reduced by each DendriticIntegrationNode
        """
        assert activations.shape[-1] == self.num_neurons
        assert (*activations.shape[:-1], self.input_size) == tuple(input_vector.shape)
        assert (activations.shape[-2], self.time_representation_dim) == tuple(t_embeddings_schedule.shape)
        
        preact_axon = self.axon_weights(activations)
        preact_time = self.time_weights(t_embeddings_schedule)
        preact_inputs = self.input_weights(input_vector)

        preact = preact_axon
        for preac_external in [preact_time, preact_inputs]:
            num_extra_dims = len(activations.shape) - len(preac_external.shape)
            preact = preact + preac_external[*[None]*num_extra_dims].expand(preact.shape)

        preact = preact.reshape(*preact.shape[:-1], self.num_neurons, *self.branching_factors)

        activation = torch.nn.functional.softplus(preact)
        return activation




class DendriticResidualModel(nn.Module):
    """
    XXX: doc 
    """
    def __init__(
        self,
        state_space_size: int,
        branching_factors: List[int],
        input_size: int,
        time_embedding_size: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.state_space_size = state_space_size
        self.time_embedding_size = time_embedding_size
        self.branching_factors = branching_factors

        assert len(branching_factors)
        tree_layers: List[DendriticIntegrationNode | DendriticIntegrationFanoutNode] = [
            DendriticIntegrationFanoutNode(state_space_size, branching_factors, time_embedding_size, input_size)
        ]
        for b in range(len(branching_factors)):
            tree_layers.insert(1, DendriticIntegrationNode(state_space_size, branching_factors[:b+1], time_embedding_size))
        self.axonal_tree: List[DendriticIntegrationFanoutNode | DendriticIntegrationNode] = nn.ModuleList(tree_layers)

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [..., T, input_size],             <s> passed to all </s>
        """
        node_activation = self.axonal_tree[0].forward(x, t_embeddings_schedule, input_vector)
        for tree_layer in self.axonal_tree[1:]:
            node_activation = tree_layer(node_activation, t_embeddings_schedule)
        return node_activation
