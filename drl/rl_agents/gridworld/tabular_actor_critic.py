from drl.rl_envs.gridworld.cell_types import TransitionInformationBatch

import torch
from torch import nn
from torch import Tensor as _T

from typing import Union, Optional, Tuple, Any
from dataclasses import dataclass


from ddpm.model.embedding_reprs import HierarchicalEmbeddingsBlock
from dynamic_observer.model import (
    LinearIncreaseNoiseSchedule, ContinuousTimeScoreMatchingDiffusionModel,
    FCScoreApproximator, EulerDiscretiser,
)


from abc import ABC, abstractmethod


@dataclass
class TD1ErrorUpdateInformation:
    """
    Everything of shape [batch, reward_dim]
    XXX: check compatability with multi-variate rewards!
    """
    previous_q: _T  # Q(s, a), size [batch, reward_dim]
    next_q: _T      # Q(s', a'), size [batch, reward_dim]
    target_q: _T    # r + gamma * Q(s', a') or r if terminal, size [batch, reward_dim]


@dataclass
class GenerativeTD1ErrorUpdateInformation:
    """
    Everything of shape [batch, samples, reward_dim] or [batch]
    XXX: check compatability with multi-variate rewards!
    """
    previous_q: _T  # Q(s, a) = mean(Z(s, a)), size [batch, reward_dim]
    next_q: _T      # Q(s', a'), size [batch, reward_dim]
    target_q: _T    # r + gamma * Q(s', a') or r if terminal, size [batch, reward_dim]
    previous_q_samples: _T  # size [batch, samples, reward_dim]
    next_q_samples: _T      # size [batch, samples, reward_dim]
    target_q_samples: _T    # size [batch, samples, reward_dim]



class GridWorldCriticModelBase(nn.Module, ABC):

    is_distributional: bool
    
    def __init__(self, reward_dim: int, num_states: int, discount_factor: float, num_actions_per_state: int) -> None:
        super().__init__()
        self.reward_dim = reward_dim    # This is typicaly reward size, but could also be taken to include SR dimensionality
        self.num_states = num_states
        self.num_actions_per_state = num_actions_per_state
        self.discount_factor = discount_factor

    @abstractmethod
    def get_q_values(self, states: _T, actions: Optional[_T] = None) -> _T:
        """
        states (and actions) of size [B, D]
        
        if actions == None: return [B, A, D]
        else: return [B, D]
        """
        raise NotImplementedError

    def get_targets(self, transition_batch: TransitionInformationBatch, next_actions: _T) -> TD1ErrorUpdateInformation:
        """
        Default to a TD1 error

        Currently Q(s, a) derived from same parameters as Q(s', a')

        transition_batch contains information about s, a, r, s'
        next_actions gives a', which may be smaller than a, given terminal transitions
        """
        start_state_values = self.get_q_values(transition_batch.old_cell_ids, transition_batch.actions) # [B, D]

        targets = transition_batch.transition_rewards   # [B, 1] XXX: make this actually multidim 
        import pdb; pdb.set_trace(header = 'check that targets here is [B, 1]')
        non_terminal_mask = ~transition_batch.terminal

        if non_terminal_mask.any():
            with torch.no_grad():
                end_state_values = self.get_q_values(transition_batch.new_cell_ids[non_terminal_mask], next_actions)    # [B non terminal, D]
                targets[non_terminal_mask] += (self.discount_factor * end_state_values)

        return TD1ErrorUpdateInformation(
            previous_q=start_state_values,
            next_q=end_state_values,
            target_q=targets
        )
    
    def get_loss(self, td1_info: TD1ErrorUpdateInformation, **kwargs) -> Tuple[_T, Optional[_T]]:
        """
        Used to perform, which is just Q <- Q + learning_rate * td1_info.td1_error for the basic case!
        """
        return 0.5 * (td1_info.target_q.detach() - td1_info.previous_q).square().mean(), None




class TabularGridWorldCriticModel(GridWorldCriticModelBase):

    is_distributional = False

    def __init__(self, reward_dim: int, num_states: int, discount_factor: float, num_actions_per_state: int = 4) -> None:
        super().__init__(reward_dim, num_states, discount_factor, num_actions_per_state)
        self.register_parameter('values', torch.nn.Parameter(6 * torch.ones(num_states, num_actions_per_state, reward_dim).float()))

    def get_q_values(self, states: _T, actions: Optional[_T] = None) -> _T:
        if actions is None:
            return self.values[states]
        else:
            return self.values[states, actions]




class GenerativeDistributionalGridWorldCriticModelBase(GridWorldCriticModelBase, ABC):

    is_distributional = True
    
    @abstractmethod
    def get_q_values(self, states: _T, actions: Optional[_T] = None, average: bool = True) -> _T:
        """
        states (and actions) of size [B]
        
        if actions == None: return [B, A, K, D]
        else: return [B, K, D]

        if average = True, mean samples axis, so [B, A, D] or [B, D]

        XXX: update for multidimensional r
        """
        raise NotImplementedError

    def get_targets(self, transition_batch: TransitionInformationBatch, next_actions: _T) -> GenerativeTD1ErrorUpdateInformation:
        """
        Default to a TD1 error, but this time with samples

        Currently Q(s, a) derived from same parameters as Q(s', a')

        transition_batch contains information about s, a, r, s'
        next_actions gives a', which may be smaller than a, given terminal transitions
        """
        start_state_value_samples = self.get_q_values(transition_batch.old_cell_ids, transition_batch.actions, average = False) # [B, K, D]

        targets = transition_batch.transition_rewards.unsqueeze(-2).repeat(1, start_state_value_samples.shape[-2], 1) # [B, D] -> [B, 1, D] -> [B, K, D]
        non_terminal_mask = ~transition_batch.terminal

        if non_terminal_mask.any():
            with torch.no_grad():
                end_state_values = self.get_q_values(transition_batch.new_cell_ids[non_terminal_mask], next_actions, average=False).detach()    # [B non-term, K, D]
                targets[non_terminal_mask] += (self.discount_factor * end_state_values)
        else:
            end_state_values = torch.zeros(0,targets.shape[1:])

        return GenerativeTD1ErrorUpdateInformation(
            previous_q_samples=start_state_value_samples,
            previous_q=start_state_value_samples.mean(-2),
            next_q_samples=end_state_values,
            next_q=end_state_values.mean(-2),
            target_q_samples=targets,
            target_q=targets.mean(-2),
        )





class FactoredEmbeddingDDPMCriticModel(GenerativeDistributionalGridWorldCriticModelBase):
    """
    DDPM takes in an embedding for state, and another for action
    Outputs a set of samples for the dstate value, 
    """

    def __init__(self, reward_dim: int, num_states: int, discount_factor: float, num_actions_per_state: int, state_action_embedding_dim: int, diffusion_time_embedding_size: int, num_generation_timesteps: int = 64, noising_batch_size: int = 256, device: str = 'cuda') -> None:
        super().__init__(reward_dim, num_states, discount_factor, num_actions_per_state)

        self.to(device)

        self.num_generation_timesteps = num_generation_timesteps
        self.noising_batch_size = noising_batch_size

        self.input_model, input_size = self.make_input_model(state_action_embedding_dim, device)
        self.diffmodel = self.make_smdm(reward_dim, input_size, diffusion_time_embedding_size)

    def make_smdm(self, sample_dim: int, input_size: int, diffusion_time_embedding_size: int) -> ContinuousTimeScoreMatchingDiffusionModel:
        
        noise_schedule = LinearIncreaseNoiseSchedule(0.2, 0.2, duration = 50)

        score_func = FCScoreApproximator(
            sample_size = sample_dim,
            hidden_layers = [16, 16, 16],
            input_tensor_size = input_size,   # Flattened mu and Sigma
            input_repr_size = input_size,
            input_hidden_layers = None,
            time_embedding_dim = diffusion_time_embedding_size,
            time_embedding_hidden_layers = None,
        )

        discretiser = EulerDiscretiser()
        
        diffmodel = ContinuousTimeScoreMatchingDiffusionModel(
            sample_dim=sample_dim, 
            noise_schedule=noise_schedule, 
            score_approximator=score_func, 
            discretiser=discretiser
        )

        return diffmodel

    def make_input_model(self, state_action_embedding_dim: int, device: str):
        return HierarchicalEmbeddingsBlock(
            time_embedding_dim = state_action_embedding_dim,
            num_embeddings = (self.num_states, self.num_actions_per_state),       # Don't want the s=-1 state to have a smooth continuation of the others
            time_stack_dim = 0,
            device = device
        ), 2 * state_action_embedding_dim


    def generate_inputs_from_states_and_actions(self, states: _T, actions: _T) -> Any:
        return states, actions

    @torch.no_grad()
    def get_q_values(self, states: _T, actions: Optional[_T] = None, average: bool = True, num_samples = 512) -> _T:
        """
        This only returns the final samples! Not the intermediary diffusion
            This means it's not good for training

        if average: [B, D] or [B, A, D]
        else: [B, K, D] or [B, A, K, D], where K is number of samples
        """
        batch_size = states.shape[0]
        if actions is None:
            actions = torch.arange(self.num_actions_per_state).to(states.device)
            states, actions = torch.meshgrid(states, actions)
            states = states.unsqueeze(-1).repeat(1, 1, num_samples)
            actions = actions.unsqueeze(-1).repeat(1, 1, num_samples)
            samples_shape = [batch_size, self.num_actions_per_state, num_samples]

        else:
            states = states.unsqueeze(-1).repeat(1, num_samples)
            actions = actions.unsqueeze(-1).repeat(1, num_samples)
            samples_shape=[batch_size, num_samples]

        network_input = self.generate_inputs_from_states_and_actions(states, actions)
        input_repr = self.input_model(network_input, self.num_generation_timesteps-1)

        start_samples = torch.randn(*samples_shape, self.reward_dim, device = states.device)

        sample_trajectory = self.diffmodel.run_unconditioned_reverse_dynamics(start_samples = start_samples, stimulus = (input_repr,), num_steps = self.num_generation_timesteps)
        samples = sample_trajectory[...,-1,:]

        if average:
            return samples.mean(-2)
        else:
            return samples

    def get_loss(self, td1_info: GenerativeTD1ErrorUpdateInformation, transition_batch: TransitionInformationBatch) -> Tuple[_T, Optional[_T]]:
        """
        Residual loss on the whole diffusion process
        Require transition batch so that we can condition on the right states and actions
        """

        num_samples = td1_info.target_q_samples.shape[-2]
        states = transition_batch.old_cell_ids.unsqueeze(-1).repeat(1, num_samples)
        actions = transition_batch.actions.unsqueeze(-1).repeat(1, num_samples)
        sa_inputs = self.generate_inputs_from_states_and_actions(states, actions)
        

        with torch.no_grad():
            noised_timepoints, noised_information = self.diffmodel.noise_schedule.random_noise(
                x0 = td1_info.target_q_samples.detach(),        # [B, K, 1]
                num_timepoints=self.noising_batch_size          # T
            )

        input_reprs = self.input_model(sa_inputs, self.noising_batch_size)

        approx_score = self.diffmodel.score_approximator.approximate_score(
            x_t = noised_information['x_t'],
            stimuli = (input_reprs,),
            t = noised_timepoints
        )

        real_score = noised_information['conditional_score']
        total_loss = (real_score - approx_score).square().mean()

        stepwise_losses = None  # XXX: fix timepoints?

        return total_loss, stepwise_losses


class UniqueEmbeddingDDPMCriticModel(FactoredEmbeddingDDPMCriticModel):

    def make_input_model(self, state_action_embedding_dim: int, device: str):
        return HierarchicalEmbeddingsBlock(
            time_embedding_dim = 2 * state_action_embedding_dim,    # For equity!
            num_embeddings = (self.num_states * self.num_actions_per_state,),       # Don't want the s=-1 state to have a smooth continuation of the others
            time_stack_dim = 0,
            device = device,
        ), 2 * state_action_embedding_dim

    def generate_inputs_from_states_and_actions(self, states: _T, actions: _T) -> Any:
        return (states * self.num_actions_per_state + actions, )



class UniqueEmbeddingDDPMCriticModelWithSuccessorRepresentationSampling(UniqueEmbeddingDDPMCriticModel):
    """
    DDPM takes in an embedding for state, and another for action
    Outputs a set of samples for the state-action value, and for the successor representation of that state

    So D = Dr + Ds - 1
        Last dimension is made up during logit transform of last Ds - 1 dimensions
    """

    def __init__(self, reward_dim: int, num_states: int, discount_factor: float, num_actions_per_state: int, state_action_embedding_dim: int, diffusion_time_embedding_size: int, device: str) -> None:

        sample_dim = reward_dim + num_states - 1

        super().__init__(
            sample_dim, num_states, discount_factor, num_actions_per_state,
            state_action_embedding_dim, diffusion_time_embedding_size, device
        )

        self.to(device)

        self.input_model, input_size = self.make_input_model(state_action_embedding_dim, device)
        self.diffmodel = self.make_smdm(sample_dim, input_size, diffusion_time_embedding_size)
    
    def get_targets(self, transition_batch: TransitionInformationBatch, next_actions: _T) -> GenerativeTD1ErrorUpdateInformation:
        """
        Default to a TD1 error, but this time with samples

        also include a successor-representation bootstrap backup

        Currently Q(s, a) and Psi(s, .) derived from same parameters as Q(s', a') and Psi(s', .)

        transition_batch contains information about s, a, r, s'
        next_actions gives a', which may be smaller than a, given terminal transitions

        XXX: update for multidimensional r
        """

        import pdb; pdb.set_trace(header = 'shapes throughout')
        
        start_state_value_samples = self.get_q_values(transition_batch.old_cell_ids, transition_batch.actions, average = False) # [B, K, D]

        return_targets = transition_batch.transition_rewards.unsqueeze(-2).repeat(1, start_state_value_samples.shape[-2], 1) # [B, Dr] -> [B, 1, Dr] -> [B, K, Dr]

        sr_targets = torch.zeros(transition_batch.transition_rewards.shape[0], self.num_states) # [B, Ds - 1]
        
        sr_targets[torch.arange(len(sr_targets)), transition_batch.new_cell_ids] = (1.0 - self.discount_factor)

        import pdb; pdb.set_trace(header = 'correct infilling? ALSO NEED TO DO LOGIT TRANSFORM HERE')
        sr_targets = sr_targets.unsqueeze(-2).repeat(1, start_state_value_samples.shape[-2], 1) # [B, K, Ds - 1]
        
        targets = torch.concat([return_targets, sr_targets], -1)

        non_terminal_mask = ~transition_batch.terminal

        if non_terminal_mask.any():
            with torch.no_grad():
                end_state_values_and_srs = self.get_q_values(transition_batch.new_cell_ids[non_terminal_mask], next_actions, average=False).detach()    # [B non-term, K, D]



                targets[non_terminal_mask] += (self.discount_factor * end_state_values_and_srs)

        else:
            end_state_values = torch.zeros(0, targets.shape[1:])

        import pdb; pdb.set_trace(header = 'check the shapes!')
        
        return GenerativeTD1ErrorUpdateInformation(
            previous_q_samples=start_state_value_samples,
            previous_q=start_state_value_samples.mean(-2),
            next_q_samples=end_state_values,
            next_q=end_state_values.mean(-2),
            target_q_samples=targets,
            target_q=targets.mean(-2),
        )







class GridWorldActorModel(nn.Module, ABC):

    requires_q_values: bool

    def __init__(self, num_states: int, num_actions_per_state: int = 4) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions_per_state = num_actions_per_state

    @staticmethod
    @torch.no_grad()
    def sample_actions(pmfs: _T) -> _T:
        cdfs = pmfs.cumsum(-1)
        u = torch.rand(len(cdfs), 1)
        choices = (u > cdfs).sum(-1)
        return choices

    @abstractmethod
    def choose_actions(self, current_states: _T, *args) -> Union[_T, _T]:
        """
        Returns probabilties over actions, then the choices made
        
        When doing eps-greedy, just pass the state action value samples here
        """
        raise NotImplementedError

    def get_loss(
        self, 
        action_probs: _T,
        action_choices: _T,
        current_state_action_values: _T
    ) -> _T:
        """
        Ascent is done on Q(s, a) * log\pi(a|s), so we return the negative

        current_state_action_values gives all action values for (s, a) which we are leaving
            shaped [B]

            Should be detached!
        
        action_probs is the same shape, generated by self.choose_actions
        """
        assert len(current_state_action_values.shape) == 1, "current_state_action_values should be fed into GridWorldActorModel.get_loss with no extra dimensionality!"

        relevant_probs = action_probs[torch.arange(len(action_probs)), action_choices]
        relevant_log_probs = relevant_probs.log()
        return - (current_state_action_values.detach() * relevant_log_probs).mean()




class TabularActorModel(GridWorldActorModel):

    requires_q_values = False

    def __init__(self, num_states: int, num_actions_per_state: int = 4) -> None:
        super().__init__(num_states, num_actions_per_state)
        self.register_parameter('logits', torch.nn.Parameter(2.0 * torch.randn(num_states, num_actions_per_state)))
    
    def choose_actions(self, current_states: _T) -> Tuple[_T, _T]:
        if len(current_states) == 0:
            return torch.tensor([]).unsqueeze(1).reshape(-1, self.num_actions_per_state), torch.tensor([])
        pmfs = self.logits[current_states].softmax(-1)
        choices = self.sample_actions(pmfs)
        return pmfs, choices



class EpsGreedyActorModel(GridWorldActorModel):

    requires_q_values = True

    def __init__(self, epsilon: float, num_states: int, num_actions_per_state: int = 4) -> None:
        super().__init__(num_states, num_actions_per_state)
        self.epsilon = epsilon
        if epsilon == 1.0:
            self.requires_q_values = False # speed it up
    
    def choose_actions(self, current_states: _T, state_value_vectors: Optional[_T] = None) -> Tuple[_T, _T]:
        """
        current_states of shape [B], indices
        state_value_vectors of shape [B, A]
        """
        pmfs = torch.ones(len(current_states), self.num_actions_per_state) * (self.epsilon / self.num_actions_per_state)
        pmfs = pmfs.to(current_states.device)
        if state_value_vectors is None:
            assert self.epsilon == 1.0, "EpsGreedyActorModel can only accept state_value_vectors == None if epsilon == 1.0"
        else:
            assert len(state_value_vectors.shape) == 1, "state_value_vectors should be fed into EpsGreedyActorModel.choose_actions with no extra dimensionality!"
            greedy_choices = state_value_vectors.argmax(-1)
            pmfs[torch.arange(state_value_vectors.shape[0]), greedy_choices] += (1-self.epsilon)
        choices = self.sample_actions(pmfs)
        return pmfs, choices

    def get_loss(
        self, 
        action_probs: _T,
        action_choices: _T,
        current_state_action_values: _T
    ) -> _T:
        return torch.tensor(0.0)

