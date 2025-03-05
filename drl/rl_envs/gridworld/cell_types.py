from __future__ import annotations

import torch
from torch import Tensor as _T

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


OPPOSITE_DIRECTIONS = {
    'N': 'S',
    'E': 'W',
    'W': 'E',
    'S': 'N',
}


@dataclass
class EntryInformation:
    """
    Return index:
        0: whether entry was succesful or not
        1: whether transition was terminal or not
        2: reward for entry
    Can have non-zero reward even if entry failed, e.g. a spike wall
    """
    success: bool
    terminal: bool
    entry_reward: float


@dataclass
class TransitionInformation:
    transition_reward: float
    action: int
    old_cell: GridCell
    new_cell: GridCell
    terminal: bool


@dataclass
class TransitionInformationBatch:
    transition_informations: List[TransitionInformation]

    def __len__(self) -> int:
        return len(self.transition_informations)

    def __post_init__(self):
        self.transition_rewards = torch.tensor([tis.transition_reward for tis in self.transition_informations]) # r
        self.old_cell_ids = torch.tensor([tis.old_cell.cell_id for tis in self.transition_informations])    # s
        self.new_cell_ids = torch.tensor([tis.new_cell.cell_id for tis in self.transition_informations])    # s'
        self.actions = torch.tensor([tis.action for tis in self.transition_informations])                   # a
        self.terminal = torch.tensor([tis.terminal for tis in self.transition_informations])
        self.num_terminal = sum(self.terminal)
        self.non_terminal_new_state_ids = torch.tensor([tis.new_cell.cell_id for tis in self.transition_informations if not tis.terminal])


class GridCell(ABC):
    
    start_probability_logit: float = - float('inf') # Logit for softmax of starting states. Default to not starting there
    can_terminate: bool # Lets gridworld class know that trial can terminate from visiting here

    def __init__(self, cell_id: int) -> None:
        self.neighbours: Dict[str, GridCell] = {} # keys one of N E W S
        self.cell_id = cell_id

    def add_neighbour(self, other: GridCell, direction: str) -> None:
        opposite_direction = OPPOSITE_DIRECTIONS[direction]
        
        assert direction not in self.neighbours, f"{direction} neighbour already assigned"
        assert opposite_direction not in other.neighbours, f"{opposite_direction} neighbour already assigned"
        
        self.neighbours[direction] = other
        other.neighbours[opposite_direction] = self
    
    def attempt_leave(self, direction: str) -> TransitionInformation:
        other_cell = self.neighbours[direction]
        from_direction = OPPOSITE_DIRECTIONS[direction]
        entry_information = other_cell.attempt_enter(from_direction)
        exit_reward = self.exit_reward(direction, entry_information.success)
        return TransitionInformation(
            transition_reward = entry_information.entry_reward + exit_reward,
            action = 'NEWS'.index(direction),
            old_cell = self,
            new_cell = (
                other_cell if entry_information.success 
                else self
            ),
            terminal = entry_information.terminal,
        )

    @abstractmethod
    def attempt_enter(self, from_direction: str) -> EntryInformation:
        """
        from_direction of one N E W S
        """
        raise NotImplementedError

    @abstractmethod
    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        """
        from_direction of one N E W S and typically doesn't matter!
        exit_success is the same as index 0 from attempt_enter
        """
        raise NotImplementedError




class NormalCorridorCell(GridCell):
    
    can_terminate = False

    def attempt_enter(self, from_direction: str) -> EntryInformation:
        return EntryInformation(True, False, 0.0)
    
    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        return 0.0
    


class NormalWallCell(GridCell):
    
    can_terminate = False

    def attempt_enter(self, from_direction: str) -> EntryInformation:
        return EntryInformation(False, False, 0.0)
    
    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        raise TypeError('Should not be here ever!')
    

class StartPointCell(NormalCorridorCell):
    
    can_terminate = False
    
    def __init__(self, cell_id: int, start_probability_logit: float) -> None:
        super().__init__(cell_id)
        self.start_probability_logit = start_probability_logit



class RewardingTerminalCell(GridCell):
    
    can_terminate = True

    def __init__(self, cell_id: int, mean: float, std: float) -> None:
        super().__init__(cell_id)
        self.mean = mean
        self.std = std
    
    def sample_reward(self):
        return np.random.randn() * self.std + self.mean
       
    def attempt_enter(self, from_direction: str) -> EntryInformation:
        return EntryInformation(True, True, self.sample_reward())

    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        raise TypeError('Should not be able to leave this!')

