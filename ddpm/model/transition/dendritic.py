import torch
from torch import nn
from torch import Tensor as _T

from typing import List

from math import prod


class DendriticIntegrationNode(nn.Module):

    """
    Represents num_nodes = N different nodes in the dendritic 'tree',
        with branching factor of branching_factor = B

    High-level calc:
        Input comes in shaped [BN]
        Reshaped to [B, N]
            Careful: adjacent elements of input are to be grouped into the same
                *columns* of this reshape
            That is: reshaped[:,n] = input[B*n:(B+1)*n]
            This means that inputs going into the same node are adjacent in the
                original tensor
        Then einsumed to [N]
    """

    def __init__(
        self,
        num_nodes: int,
        branching_factor: int,
        time_representation_dim: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.branching_factor = branching_factor
        self.time_representation_dim = time_representation_dim
        self.branched_size = num_nodes * branching_factor
        self.bias = bias

        self.register_parameter('propagation_weights', torch.nn.Parameter(torch.zeros(branching_factor, num_nodes)))
        nn.init.xavier_normal_(self.propagation_weights)

        self.time_representation_weights = nn.Linear(time_representation_dim, num_nodes, bias = bias)

    def preactivation(self, activations: _T, t_embeddings_schedule: _T) -> List[_T]:
        assert activations.shape[-1] == self.branched_size, \
            f"Got activations of shape {activations.shape}, expected it to end with {self.branched_size} = B ({self.branching_factor}) x N ({self.num_nodes})"
        assert (activations.shape[-2], self.time_representation_dim) == tuple(t_embeddings_schedule.shape)
        reshaped_activations = activations.reshape(*activations.shape[:-1], self.num_nodes, self.branching_factor).transpose(-1, -2)
        # -> assert activations[...,:self.branching_factor] == reshaped_activations[...,:,0]
        preact_dendrite = torch.einsum('...bn,bn->...n', reshaped_activations, self.propagation_weights)
        return preact_dendrite, self.time_representation_weights(t_embeddings_schedule)

    def forward(self, activations: _T, *other_inputs: _T) -> _T:
        """
        Expecting:
            activations shaped [..., T, num_nodes * branching_factor]
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

    def __init__(self, num_neurons: int, branching_factors: List[int], time_representation_dim: int, input_size: int) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.num_nodes = num_neurons * prod(branching_factors)
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
        Output of shape [..., num_neurons * branching_factor]
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
        for bf in branching_factors:
            tree_layers.append(DendriticIntegrationNode(int(tree_layers[-1].num_nodes / bf), bf, time_embedding_size))
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

