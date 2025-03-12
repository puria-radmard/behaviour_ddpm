from __future__ import annotations

import torch
from torch import Tensor as _T

import numpy as np

import random

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
        success: whether entry was succesful or not
        terminal: whether transition was terminal or not
        entry_reward: reward for entry
    Can have non-zero reward even if entry failed, e.g. a spike wall
    """
    resulting_cell: GridCell
    terminal: bool
    entry_reward: float


@dataclass
class TransitionInformation:
    transition_reward: List[float]
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

    def __init__(self, cell_id: int, coords: Tuple[int]) -> None:
        self.neighbours: Dict[str, GridCell] = {} # keys one of N E W S
        self.cell_id = cell_id
        self.coords = coords

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
        success = (entry_information.resulting_cell == other_cell)
        exit_reward = self.exit_reward(direction, success)
        return TransitionInformation(
            transition_reward = [entry_information.entry_reward + exit_reward], # By default dim 1
            action = 'NEWS'.index(direction),
            old_cell = self,
            new_cell = entry_information.resulting_cell,
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
        return EntryInformation(self, False, 0.0)
    
    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        return 0.0
    


class NormalWallCell(GridCell):
    
    can_terminate = False

    def attempt_enter(self, from_direction: str) -> EntryInformation:
        return EntryInformation(self.neighbours[from_direction], False, 0.0)
    
    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        raise TypeError('Should not be here ever!')


class SpikyWallCell(GridCell):
    
    can_terminate = False

    def attempt_enter(self, from_direction: str) -> EntryInformation:
        return EntryInformation(self.neighbours[from_direction], False, -1.0)
    
    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        raise TypeError('Should not be here ever!')


class WindyValveCell(GridCell):
    
    can_terminate = False

    def __init__(self, cell_id: int, coords: Tuple[int], valve_side: str, spit_sides: str) -> None:
        super().__init__(cell_id, coords)
        assert (valve_side in 'NEWS') and (len(valve_side) == 1)
        assert set(spit_sides).issubset(set('NEWS'))
        self.valve_side = valve_side
        self.spit_sides = spit_sides

    def attempt_enter(self, from_direction: str) -> EntryInformation:
        if from_direction == self.valve_side:
            randomly_chosen_cell = random.choice(self.spit_sides)
            return EntryInformation(self.neighbours[randomly_chosen_cell], False, -1.0)

        else:
            return EntryInformation(self.neighbours[from_direction], False, -1.0)


    

class StartPointCell(NormalCorridorCell):
    
    can_terminate = False
    
    def __init__(self, cell_id: int, coords: Tuple[int], start_probability_logit: float) -> None:
        super().__init__(cell_id, coords)
        self.start_probability_logit = start_probability_logit



class RewardingTerminalCell(GridCell):
    
    can_terminate = True

    def __init__(self, cell_id: int, coords: Tuple[int], mean: float, std: float) -> None:
        super().__init__(cell_id, coords)
        self.mean = mean
        self.std = std
    
    def sample_reward(self):
        return np.random.randn() * self.std + self.mean
       
    def attempt_enter(self, from_direction: str) -> EntryInformation:
        return EntryInformation(self, True, self.sample_reward())

    def exit_reward(self, to_direction: str, exit_success: bool) -> float:
        raise TypeError('Should not be able to leave this!')


class BimodalRewardingTerminalCell(RewardingTerminalCell):

    def __init__(self, cell_id: int, coords: Tuple[int], mean1: float, std1: float, pi1: float, mean2: float, std2: float, pi2: float) -> None:
        super(RewardingTerminalCell, self).__init__(cell_id, coords)
        self.mean1 = mean1
        self.std1 = std1
        self.pi1 = pi1
        self.mean2 = mean2
        self.std2 = std2
        self.pi2 = pi2
        assert self.pi1 + self.pi2 == 1.0
        self.means = [mean1, mean2]
        self.stds = [std1, std2]
        
    def sample_reward(self):
        mode = int(np.random.rand() < self.pi1)
        return np.random.randn() * self.stds[mode] + self.means[mode]


