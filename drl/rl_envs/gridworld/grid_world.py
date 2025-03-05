import torch
from torch import Tensor as _T

import numpy as np

from typing import Dict, List, Tuple, Type, Any

from drl.rl_envs.gridworld.cell_types import GridCell, TransitionInformationBatch

from drl.rl_envs.gridworld.cell_types import (
    NormalCorridorCell,
    NormalWallCell,
    StartPointCell,
)

from matplotlib.font_manager import FontProperties

import seaborn as sns
from matplotlib.pyplot import Axes




STANDARD_GRID_CONFIG: Dict[str, Tuple[Type[GridCell], Dict[str, Any]]] = {
    '.': (NormalCorridorCell, {}),
    '#': (NormalWallCell, {}),
    'S': (StartPointCell, {'start_probability_logit': 0.0}),

}


class GridWorld:

    """
    grid_text should be rectangular, e.g.:

    ########
    #S.....#
    #......#
    #.....H#
    #......#
    #..L...#
    ########

    As with STANDARD_GRID_CONFIG, grid_config maps from characters in grid_text to
        string names of GridCell subclasses in gridworld.cell_types, and associated kwargs
    """

    cells: List[GridCell]   # Should not be tampered... change to tuple?

    def __init__(self, grid_text: str, grid_config: Dict[str, Tuple[Type[GridCell], Dict[str, Any]]]) -> None:
        
        if grid_text.startswith('\n'):
            grid_text = grid_text[1:]
        if grid_text.endswith('\n'):
            grid_text = grid_text[:-1]
        all_grid_config = STANDARD_GRID_CONFIG.copy()
        all_grid_config.update(grid_config)

        all_rows = grid_text.split('\n')
        self.height = len(all_rows)
        self.width = len(all_rows[0])
        self.grid_text = np.array([list(row) for row in all_rows])

        # Generate grid
        self.cells = []
        
        for r_idx, row in enumerate(all_rows):
            
            assert len(row) == self.width, 'Cannot have non-rectangular gridworld for now!'

            for c_idx, char in enumerate(row):

                cell_class, cell_kwargs = all_grid_config[char]
                new_cell = cell_class(cell_id=len(self.cells), **cell_kwargs)

                if c_idx > 0:
                    new_cell.add_neighbour(self.cells[-1], 'W')
                
                if c_idx == self.width - 1:
                    new_cell.add_neighbour(self.cells[-self.width+1], 'E')

                if r_idx > 0:
                    new_cell.add_neighbour(self.cells[-self.width], 'N')

                self.cells.append(new_cell)

        for start_idx, end_idx in enumerate(range(-self.width, 0)):
            self.cells[start_idx].add_neighbour(self.cells[end_idx], 'N')
        
        assert all([len(cell.neighbours) == 4 for cell in self.cells])
        assert any([cell.can_terminate for cell in self.cells])
        
        self.starting_logits = torch.tensor([cell.start_probability_logit for cell in self.cells])
        starting_probs = torch.softmax(self.starting_logits, dim = 0)
        assert not torch.isinf(max(self.starting_logits)) and not torch.isnan(starting_probs).any()
        self.starting_probs = torch.distributions.Categorical(starting_probs)
    
    def start_trials(self, num_trials: int) -> _T:
        """
        The most basic case now, where there are a set number of starting states,
            each with a logit of starting there
        
        Just taking samples from the starting_probs tensor
        """
        return self.starting_probs.sample([num_trials])
    
    def take_step(self, actions: _T, current_states: _T) -> TransitionInformationBatch:
        """
        actions are fed in as indices, which index 'NEWS'
        current states is an index of (int) cell_id's
        """
        assert actions.shape == current_states.shape and len(current_states.shape) == 1
        return TransitionInformationBatch([
            self.cells[s].attempt_leave('NEWS'[a])
            for a, s in zip(actions, current_states)
        ])
    
    def fill_in_trials(self, previous_transitions: TransitionInformationBatch, target_batch_size: int) -> Tuple[_T, int]:
        """
        Takes in information from previous_transitions.non_terminal_new_state_ids,
            and initialises enough new trials to give back a tensor of state ids of length target_batch_size
        
        These state ids can then be fed to take_step
        """
        existing_traj_new_states = previous_transitions.non_terminal_new_state_ids
        num_new_trials = target_batch_size - existing_traj_new_states.shape[0]
        if num_new_trials > 0:
            new_trials_states = self.start_trials(num_new_trials)
            return torch.concat([existing_traj_new_states, new_trials_states], dim = 0)
        else:
            return existing_traj_new_states

    def display_q_values(self, q_values: _T, axes: Axes) -> None:
        assert len(axes) == 4 == q_values.shape[1]
        assert len(q_values.shape) == 2
        reshaped_q_values = q_values.reshape(self.height, self.width, 4)
        wall_mask = (self.grid_text == '#')
        text = self.grid_text
        vmin = q_values[~ np.isnan(q_values)].min()
        vmax = q_values[~ np.isnan(q_values)].max()
        for a_i, a_name in enumerate('NEWS'):
            action_values = reshaped_q_values[:,:,a_i]
            action_values[wall_mask] = np.nan
            sns.heatmap(
                action_values, annot = text, ax = axes[a_i], 
                vmin = vmin, vmax = vmax, fmt=''
            )
            axes[a_i].set_title(a_name)

    
    # def take_step(self, actions: _T, current_states: _T, 
    #     # target_replenish_batch: int
    #     ) -> Tuple[_T, TransitionInformationBatch]:
    #     """
    #     actions are fed in as indices, which index 'NEWS'
    #     current states is an index of (int) cell_id's

    #     1. take a step in the environment for each trial in the batch -> TransitionInformationBatch
    #     2. fill trials back in until its sized target_replenish_batch

    #     returns the new states, and the number of trials that terminated in this transition
    #     """
    #     new_transitions = self.take_step_inner(actions, current_states)
    #     new_states = new_transitions.non_terminal_new_state_ids
    #     # new_states = self.fill_in_trials(new_transitions, target_replenish_batch)
    #     return new_states, new_transitions

