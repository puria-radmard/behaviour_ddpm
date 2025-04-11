import torch
from torch import nn
from torch import Tensor as _T

from typing import List


class AxonalBranchingBlock(nn.Module):
    """
    Block for num_nodes different branching points,
    which each individually branch out to branching_factor further nodes
    """
    def __init__(self, num_nodes: int, branching_factor: int, time_representation_dim: int) -> None:
        super().__init__()
        
        self.num_nodes = num_nodes
        self.branching_factor = branching_factor
        self.time_representation_dim = time_representation_dim
        self.output_size = num_nodes * branching_factor

        # print('XXX DOING DENSE DENTRITIC TREES - I.E. A FULL NN WITH HIDDEN LAYERS')
        # XXX: self.register_parameter('propagation_weights', torch.nn.Parameter(torch.zeros(num_nodes, branching_factor * num_nodes)))

        self.register_parameter('propagation_weights', torch.nn.Parameter(torch.zeros(num_nodes, branching_factor)))
        self.register_parameter('time_representation_weights', torch.nn.Parameter(torch.zeros(time_representation_dim, num_nodes * branching_factor)))
        nn.init.xavier_normal_(self.propagation_weights)
        nn.init.xavier_normal_(self.time_representation_weights)

    def preactivation(self, activations: _T, t_embeddings_schedule: _T) -> List[_T]:
        assert activations.shape[-1] == self.num_nodes
        assert (activations.shape[-2], self.time_representation_dim) == tuple(t_embeddings_schedule.shape)
        preact_axon = torch.einsum('...n,nb->...nb', activations, self.propagation_weights)                 # n = nodes, b = branching factor
        # XXX: preact_axon = torch.einsum('...n,nb->...b', activations, self.propagation_weights).unsqueeze(-1)                 # n = nodes, b = branching factor * num nodes              XXX: see above! 
        preact_time = torch.einsum('td,dm->tm', t_embeddings_schedule, self.time_representation_weights)        # t = num timesteps, d = time embedding dim, m = nb = branching factor * num nodes
        return preact_axon, preact_time

    def forward(self, activations: _T, *other_inputs: _T) -> _T:
        """
        Expecting:
            activations shaped [..., T, num_nodes]
            t_embeddings_schedule of shape [T, time_emb_size]
            if given:
                input_vector of shape [..., T, input_size]
        Output of shape [..., num_nodes * branching_factor]
        """
        preact_axon, *preac_externals = self.preactivation(activations, *other_inputs)
        preact = preact_axon.reshape(*preact_axon.shape[:-2], self.num_nodes * self.branching_factor)
        for preac_external in preac_externals:
            num_extra_dims = len(activations.shape) - len(preac_external.shape)
            preact = preact + preac_external[*[None]*num_extra_dims].expand(preact.shape)
        activation = torch.nn.functional.leaky_relu(preact)
        return activation


class AxonalBranchingBlockWithInput(AxonalBranchingBlock):

    def __init__(self, num_nodes: int, branching_factor: int, time_representation_dim: int, input_size: int) -> None:
        super().__init__(num_nodes, branching_factor, time_representation_dim)

        self.input_size = input_size
        self.register_parameter('input_weights', torch.nn.Parameter(torch.zeros(input_size, num_nodes * branching_factor)))
        nn.init.xavier_normal_(self.input_weights)

    def preactivation(self, activations: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        assert (*activations.shape[:-1], self.input_size) == tuple(input_vector.shape)
        preact_axon, preact_time = super().preactivation(activations, t_embeddings_schedule)
        preact_input = torch.einsum('...d,dm->...m', input_vector, self.input_weights)        # t = num timesteps, d = time embedding dim, m = nb = branching factor * num nodes
        return preact_axon, preact_time, preact_input



class AxonalResidualModel(nn.Module):
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
        tree_layers: List[AxonalBranchingBlockWithInput | AxonalBranchingBlock] = [AxonalBranchingBlockWithInput(state_space_size, branching_factors[0], time_embedding_size, input_size)]
        for bf in branching_factors[1:]:
            tree_layers.append(AxonalBranchingBlock(tree_layers[-1].output_size, bf, time_embedding_size))
        self.axonal_tree: List[AxonalBranchingBlockWithInput | AxonalBranchingBlock] = nn.ModuleList(tree_layers)

        self.recombination_block = nn.Sequential(
            nn.Linear(tree_layers[-1].output_size, self.state_space_size, bias = True),
            nn.Softplus(),
        )

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [..., T, state_space_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [..., T, input_size],             <s> passed to all </s>
        """
        node_activation = self.axonal_tree[0].forward(x, t_embeddings_schedule, input_vector)
        for tree_layer in self.axonal_tree[1:]:
            node_activation = tree_layer(node_activation, t_embeddings_schedule)
        recombination = self.recombination_block(node_activation)
        return recombination

