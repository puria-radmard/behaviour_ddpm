from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Deque, Optional, Union
from matplotlib.pyplot import Axes

from matplotlib import pyplot as plt

import math
import torch
import numpy as np

from torch import Tensor as _T
from numpy import ndarray as _A


from dataclasses import dataclass

@dataclass
class SwapSampleInformation:
    sample_set: _T
    item_indices: _T



class ExampleSampleGenerator(ABC):
    """
    Can generate inputs as tabular data or as target samples

    Samples are tensor with shape [num_samples, *self.sample_shape], indexed by 'samples'
    Also output some sample metadata

    Because it contains information about the sample space, it must also be able to process samples,
        alongside the report dimension features provided as stimuli, and determine the rate of swapping to each item.
        Also determine other relevatn diagnostics
    See child class docstrings for specific examples

    Similarly, needs to be able to display (some) example samples
    """
    sample_shape: List[int]
    required_task_variable_keys: Set[str]
    diagnostics_memory = 100

    @abstractmethod
    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        raise NotImplementedError

    @abstractmethod
    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T], recent_sample_diagnostics: Deque[_A], axes: Optional[Axes] = None) -> Tuple[_T, Dict[str, _T]]:
        raise NotImplementedError

    @abstractmethod
    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes) -> None:
        raise NotImplementedError


class TabularExampleSampleGenerator(ExampleSampleGenerator):
    """
    Just generates Cartesian samples (on a point)

    Swapping determined by distance on Cartesian plane
    Only diagnostic is that magnitude of output samples ~= 1
    """
    required_task_variable_keys = {'report_features_cart', 'swap_probabilities'}
    sample_shape = [2]

    def __init__(self) -> None:
        pass

    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        selection_pmf = variable_dict['swap_probabilities']
        selected_item_idx = torch.tensor(np.random.choice(selection_pmf.shape[0], size = num_samples, replace = True, p = selection_pmf))
        selected_cartesian = variable_dict['report_features_cart'][selected_item_idx] 
        selected_cartesian = selected_cartesian + torch.randn_like(selected_cartesian) * 0.05
        return SwapSampleInformation(selected_cartesian, selected_item_idx)

    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T]) -> Tuple[_T, Dict[str, _T]]:
        import pdb; pdb.set_trace(header = 'get pmf by taking closest to report_feature')
        return closest_rate, {'magnitude': magnitudes}

    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes) -> None:
        if isinstance(sample_set, SwapSampleInformation):
            axes.scatter(sample_set.sample_set[:, 0], sample_set.sample_set[:, 1], c = sample_set.item_indices, alpha=0.5, s=1)
        else:
            sample_set = sample_set.cpu().numpy()
            axes.scatter(sample_set[:, 0], sample_set[:, 1], alpha=0.5, s=1)
        axes.add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))



class WhiteNoiseStripExampleSampleGenerator(ExampleSampleGenerator):
    """
    Generates single channel image of mostly mean -1 white noise
    
    From the center pointing to the desired feature dimension output, there is a mean 1 white noise
    """

    required_task_variable_keys = {'report_features', 'swap_probabilities'}

    def __init__(self, image_size: int = 256, strip_pixel_width: int = 20) -> None:
        self.sample_shape = [image_size, image_size]
        self.strip_pixel_width = strip_pixel_width

    @staticmethod
    def generate_strip_image(image_size: int, strip_pixel_width: int, selected_item_ang: torch.Tensor) -> torch.Tensor:
        """
        Claude 3.5 Sonnet

        Generate a batch of square images with strips of 1.0 values at specified angles, -1.0 elsewhere.
        
        Args:
            image_size: Size of the square image (width and height)
            strip_pixel_width: Width of the strip in pixels
            selected_item_ang: Tensor of shape [batch_size] containing angles in radians
                            (0 = horizontal right, π/2 = vertical up)
        
        Returns:
            torch.Tensor: Image tensor of shape [batch_size, image_size, image_size] 
                        with values 1.0 in strips, -1.0 elsewhere
        """
        batch_size = selected_item_ang.shape[0]
        device = selected_item_ang.device
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_size, device=device),
            torch.arange(image_size, device=device),
            indexing='ij'
        )
        
        # Center the coordinates
        x_centered = x_coords - image_size // 2
        y_centered = y_coords - image_size // 2
        
        # Calculate the angle of each pixel relative to center
        angles = torch.atan2(y_centered, x_coords - image_size // 2)
        
        # Normalize angles to be between 0 and 2π
        angles = (angles + 2 * math.pi) % (2 * math.pi)
        
        # Normalize selected angles and reshape for broadcasting
        selected_item_ang = (selected_item_ang + 2 * math.pi) % (2 * math.pi)
        selected_item_ang = selected_item_ang.view(batch_size, 1, 1)
        
        # Calculate angular difference from the selected angles
        angle_diff = torch.abs(angles.unsqueeze(0) - selected_item_ang)
        # Handle wrap-around case
        angle_diff = torch.minimum(angle_diff, 2 * math.pi - angle_diff)
        
        # Convert desired strip width to radians
        radii = torch.sqrt(x_centered**2 + y_centered**2)
        # Avoid division by zero at center
        radii = torch.maximum(radii, torch.tensor(1.0, device=device))
        angular_width = strip_pixel_width / radii
        
        # Create the batched images: 1.0 where angle difference is within strip width, -1.0 elsewhere
        images = torch.where(angle_diff <= angular_width.unsqueeze(0)/2, 1.0, -1.0)
        
        return images

    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        selection_pmf = variable_dict['swap_probabilities']
        selected_item_idx = torch.tensor(np.random.choice(selection_pmf.shape[0], size = num_samples, replace = True, p = selection_pmf))
        selected_item_ang = torch.tensor(variable_dict['report_features'])[selected_item_idx]

        noise_offset = self.generate_strip_image(image_size=self.sample_shape[0], strip_pixel_width=self.strip_pixel_width, selected_item_ang=selected_item_ang)
        
        white_noise_images = noise_offset + torch.randn_like(noise_offset)
        return SwapSampleInformation(white_noise_images, selected_item_idx)

    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T]) -> Tuple[_T, Dict[str, _T]]:
        import pdb; pdb.set_trace(header = 'get pmf by taking closest to report_feature')
        return closest_rate, {'magnitude': magnitudes}

    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes) -> None:
        if isinstance(sample_set, SwapSampleInformation):
            axes.imshow(sample_set.sample_set[0].cpu().numpy(), cmap = 'grey', vmin = -4.5, vmax = + 4.5)
        else:
            axes.imshow(sample_set[0].cpu().numpy(), cmap = 'grey', vmin = -4.5, vmax = + 4.5)
