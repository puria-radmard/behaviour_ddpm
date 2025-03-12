from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Tuple, List, Optional, Type

from abc import ABC, abstractmethod

from dynamic_observer.model.score_approximator_dispatcher import ScoreApproximatorDispatcher, TrueScoreApproximatorDispatcher
from dynamic_observer.model.noise_schedule import ContinuousTimeNoiseSchedule



class ScoreApproximator(nn.Module, ABC):

    @abstractmethod
    def approximate_score(self, x_t: _T, stimuli: Tuple[_T], t: _T, **kwargs):
        """
        x_t comes in shape [..., D]
        t comes in all ones, with same number of axes
        """
        raise NotImplementedError

    def prepare_dispatcher(self, stimuli: Tuple[_T], t: _T) -> ScoreApproximator | ScoreApproximatorDispatcher:
        return self


class TrueScore(ScoreApproximator):
    """
    By giving this `approximator` access to the true data distribution and the
    noise schedule, it can find the marginal distribution, and therefore score, of the noised
    data

    TODO: above - for now it just runs to the same fixed point everytime!
    """

    def __init__(self, noise_schedule: ContinuousTimeNoiseSchedule) -> None:
        super().__init__()
        self.noise_schedule = noise_schedule

    def prepare_dispatcher(self, stimuli: Tuple[_T, ...], t: _T) -> ScoreApproximator | ScoreApproximatorDispatcher:
        return TrueScoreApproximatorDispatcher(stimuli, t, self.noise_schedule)

    def approximate_score(self, x_t: _T, stimuli: Tuple[_T, ...], t: _T, **kwargs):
        raise Exception("Should not be directly accessed anymore!")
        reshaped_m_x0, reshaped_S_x0 = stimuli
        assert len(reshaped_m_x0.shape) == 2 and len(reshaped_S_x0.shape) == 3
        reshaped_m_x0 = reshaped_m_x0.unsqueeze(1)
        reshaped_S_x0 = reshaped_S_x0.unsqueeze(1)
        marginal_moments = self.noise_schedule.marginal_moments_gaussian_gt_distribution(reshaped_m_x0, reshaped_S_x0, t)
        m_xt = marginal_moments['m_xt']
        S_xt = marginal_moments['S_xt']
        return self.noise_schedule.marginal_score(m_xt, S_xt, x_t)


class FCScoreApproximator(ScoreApproximator):
    """
    Fully connected feedforward networkw hich approximates the conditional score of the noising process
    """
    def __init__(
        self,
        sample_size: int,
        hidden_layers: List[int],
        input_tensor_size: int,             # XXX: extend to multiple inputs?
        input_repr_size: int,
        input_hidden_layers: List[int],
        time_embedding_dim: int,
        time_embedding_hidden_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.sample_size = sample_size
        self.hidden_layers = hidden_layers
        self.input_tensor_size = input_tensor_size
        self.input_repr_size = input_repr_size
        self.input_hidden_layers = input_hidden_layers
        self.time_embedding_dim = time_embedding_dim
        self.time_embedding_hidden_layers = time_embedding_hidden_layers

        # Input representation
        self.input_layers = self.generate_fc_sequential(
            input_size = input_tensor_size,
            output_size = input_repr_size,
            hidden_layers = input_hidden_layers,
            non_linearity = nn.SiLU
        )

        # Time representation body maps from trig representations of time to an actual
        # time embedding, which is passed onto the main body
        if time_embedding_hidden_layers is None:
            time_embedding_hidden_layers = [time_embedding_dim]
        
        self.time_layers = self.generate_fc_sequential(
            input_size = time_embedding_dim,
            output_size = time_embedding_dim,
            hidden_layers = time_embedding_hidden_layers,
            non_linearity = nn.SiLU
        )

        # Main approximator body maps noised samples, input representation, and time to a score
        self.main_layers = self.generate_fc_sequential(
            input_size = sample_size + time_embedding_dim + input_repr_size,
            output_size = sample_size,
            hidden_layers = hidden_layers,
            non_linearity = nn.SiLU,
        )

        # Helper for time embedding
        assert time_embedding_dim % 2 == 0, "time embedding must be divisible by 2."
        factor = (
            2
            * torch.arange(
                start=0, end=1 + self.time_embedding_dim // 2, dtype=torch.float32,
            )[1:]
            / self.time_embedding_dim
        )
        self.register_buffer('factor', factor)  # shaped [time_embedding_dim / 2]

    @staticmethod
    def generate_fc_sequential(input_size: int, output_size: int, hidden_layers: List[int], non_linearity: Type[nn.Module]) -> nn.Sequential:
        main_layers = [nn.Linear(input_size, hidden_layers[0]), non_linearity()]
        for h_in, h_out in zip(hidden_layers[:-1], hidden_layers[1:]):
            main_layers.extend([nn.Linear(h_in, h_out), non_linearity()])
        main_layers.append(nn.Linear(hidden_layers[-1], output_size))
        main_layers = nn.Sequential(*main_layers)
        return main_layers

    def generate_time_embedding(self, t: _T) -> _T:
        """
        t comes in any shape [...], returns shape [..., time_embedding_dim]
        """
        t_reshape = t.unsqueeze(-1)
        factor = self.factor[*[None]*len(t.shape)]
        times = t_reshape / factor
        time_embs = torch.cat([torch.sin(times), torch.cos(times)], dim=-1)
        return self.time_layers(time_embs)

    def generate_main_input(self, x: _T, input_repr: _T, time_embedding: _T) -> _T:
        import pdb; pdb.set_trace(header = 'make shapes work!')
        network_input = torch.concat(
            [x_reshaped, input_repr_reshaped, time_embedding_reshaped],
            dim = -1
        )
        return network_input

    def approximate_score(self, x_t: _T, stimuli: Tuple[_T, ...], t: _T, **kwargs):
        """
        x_t of shape [..., 1, D_sample]
        stimuli length 1, of shape [..., D_stim]
        t of shape [1 * len(x_t.shape)]
        """
        assert len(stimuli) == 1, "Cannot have tuple stimuli for FCScoreApproximator yet!"
        import pdb; pdb.set_trace(header = 'make shapes work!')

        time_repr = self.generate_time_embedding(t)
        input_repr = self.input_layers(stimuli[0])
        main_input = self.generate_main_input(x_t, input_repr, time_repr)
        return self.main_layers(main_input)
        



