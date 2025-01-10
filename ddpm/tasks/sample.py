from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Deque, Optional, Union
from matplotlib.pyplot import Axes

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

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
    Can generate inputs as vectoral data or as target samples

    Samples are tensor with shape [num_samples, *self.sample_shape], indexed by 'samples'
    Also output some sample metadata

    Because it contains information about the sample space, it must also be able to process samples,
        alongside the report dimension features provided as stimuli, and determine the rate of swapping to each item.
        Also determine other relevatn diagnostics
    See child class docstrings for specific examples

    Similarly, needs to be able to display (some) example samples

    Finally, also handles the residual loss, which is just standard, unscaled MSE in most cases
    """
    sample_shape: List[int]
    required_task_variable_keys: Set[str]
    diagnostics_memory = 100
    
    task_metadata = {}

    @abstractmethod
    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        raise NotImplementedError

    @abstractmethod
    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T], recent_sample_diagnostics: Deque[_A], axes: Optional[Axes] = None) -> Tuple[_T, Dict[str, _T]]:
        raise NotImplementedError

    @abstractmethod
    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def display_early_x0_pred_timeseries(self, sample_set: _T, axes: Axes) -> None:
        raise NotImplementedError
    
    def mse(self, epsilon_hat: _T, epsilon: _T):
        """
        Both coming in shape [..., T, <sample shape>]
        """
        assert epsilon.shape == epsilon_hat.shape
        batch_size, T = epsilon_hat.shape[:2]
        *extra_dims, T = epsilon_hat.shape[:-len(self.sample_shape)]
        mse = torch.square(epsilon_hat - epsilon).reshape(*extra_dims, T, -1).mean(-1)   # [..., T]
        # scaled_mse = self.mse_scaler_schedule[*[None]*extra_dims] * mse[...,1:] # [..., T-1]
        return mse



class VectoralEmbeddedExampleSampleGenerator(ExampleSampleGenerator):
    """
    Just generates Cartesian samples (on a point)

    Swapping determined by distance on Cartesian plane
    Only diagnostic is that magnitude of output samples ~= 1
    """
    required_task_variable_keys = {'report_features_cart', 'swap_probabilities'}

    def __init__(self, sample_size: int, sample_radius: float, residual_in_behaviour_plane_only: bool, device = 'cuda') -> None:
        
        self.sample_shape = [sample_size]
        sample_space_size = 2
        self.sample_radius = sample_radius
        self.residual_in_behaviour_plane_only = residual_in_behaviour_plane_only

        if self.residual_in_behaviour_plane_only:
            raise Exception('are you sure you want to do this and not use LinearSubspaceTeacherForcedDDPMReverseProcess...?')
        
        if sample_size == sample_space_size:
            self.linking_matrix = torch.eye(sample_size).cpu().numpy()   # [dim y, dim x]
            self.linking_matrix_kernel = torch.tensor([]).cpu().numpy()   # []
            self.rotation_matrix = torch.eye(sample_size).cpu().numpy()   # []
        elif sample_size > sample_space_size:
            gaus = torch.randn(sample_size, sample_size)
            svd = torch.linalg.svd(gaus)
            orth = svd[0] @ svd[2]
            self.linking_matrix = orth[:sample_space_size].cpu().numpy()   # [dim y, dim x]
            self.linking_matrix_kernel = orth[sample_space_size:].cpu().numpy()   # [dim x - dim y, dim x]
            self.rotation_matrix = orth.cpu().numpy()   # [dim x, dim x]
            # raise Exception('are you sure you want to do this and not use LinearSubspaceTeacherForcedDDPMReverseProcess...?')
        else:
            raise NotImplementedError
        
        self.linking_matrix_mse = torch.tensor(self.linking_matrix).to(device)

        self.task_metadata = {
            'linking_matrix': self.linking_matrix,
            'linking_matrix_kernel': self.linking_matrix_kernel,
            'rotation_matrix': self.rotation_matrix,
            'sample_radius': sample_radius,
            'residual_in_behaviour_plane_only': residual_in_behaviour_plane_only
        }

    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        selection_pmf = variable_dict['swap_probabilities']                             # [batch, num items]
        batch_size = selection_pmf.shape[0]

        selected_item_idx = torch.multinomial(selection_pmf, num_samples, replacement=True)                     # [batch, sample]

        batch_idx = torch.arange(batch_size, device=variable_dict['report_features_cart'].device)[:, None].expand(batch_size, num_samples)   # [batch, sample]
        selected_cartesian = variable_dict['report_features_cart'][batch_idx, selected_item_idx]                 # [batch, sample, 2]
        selected_cartesian = selected_cartesian @ self.linking_matrix
        selected_cartesian = selected_cartesian + torch.randn_like(selected_cartesian) * 0.05
        selected_cartesian = selected_cartesian * self.sample_radius

        return SwapSampleInformation(selected_cartesian, selected_item_idx)

    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T]) -> Tuple[_T, Dict[str, _T]]:
        import pdb; pdb.set_trace(header = 'get pmf by taking closest to report_feature')
        return closest_rate, {'magnitude': magnitudes}

    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes, label = None) -> None:
        if isinstance(sample_set, SwapSampleInformation):
            samples = sample_set.sample_set[0] @ self.linking_matrix.T
            c = sample_set.item_indices[0]
        else:
            samples = sample_set[0].cpu().numpy() @ self.linking_matrix.T
            c = None
        axes.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1, c=c, label = label)
        axes.add_patch(plt.Circle((0, 0), self.sample_radius, color='red', fill = False))
    
    def display_early_x0_pred_timeseries(self, early_preds_set: _T, axes: Axes, cmap: ScalarMappable) -> None:
        """
        early_preds_set of shape [B, T, <dim x>] but in reversed order (i.e. T --> 1)
        """
        T = early_preds_set.shape[-2]
        for h in range(T):
            if T % 10 == 0:
                color = cmap.to_rgba(T - h)
                timestep_preds = early_preds_set[0,...,h,:] @ self.linking_matrix.T
                axes.scatter(timestep_preds[:, 0], timestep_preds[:, 1], alpha=0.5, s=1, color=color)
        axes.add_patch(plt.Circle((0, 0), self.sample_radius, color='red', fill = False))
    
    def mse(self, epsilon_hat: _T, epsilon: _T):
        if self.residual_in_behaviour_plane_only:
            epsilon = epsilon @ self.linking_matrix_mse.T
            epsilon_hat = epsilon_hat @ self.linking_matrix_mse.T
        return super(VectoralEmbeddedExampleSampleGenerator, self).mse(epsilon_hat, epsilon)



class RadialVectoralEmbeddedExampleSampleGenerator(VectoralEmbeddedExampleSampleGenerator):
    """
    Same as above except samples are generated from a circular Gaussian with std = sample_radius and fixed angle width
    """

    required_task_variable_keys = {'report_features', 'swap_probabilities'}

    def __init__(self, sample_size: int, sample_radius: float, residual_in_behaviour_plane_only: bool, device='cuda') -> None:
        super().__init__(sample_size, sample_radius, residual_in_behaviour_plane_only, device)

    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        """
        TODO: do this with a chi2 distribution instead
        """

        selection_pmf = variable_dict['swap_probabilities']                             # [batch, num items]
        batch_size = selection_pmf.shape[0]

        selected_item_idx = torch.multinomial(selection_pmf, num_samples, replacement=True)                     # [batch, sample]
        batch_idx = torch.arange(batch_size, device=variable_dict['report_features'].device)[:, None].expand(batch_size, num_samples)   # [batch, sample]

        sample_abs = self.sample_radius * torch.randn(batch_size, num_samples, 2, device = batch_idx.device).square().sum(-1).sqrt() # [batch, sample]
        selected_angles = variable_dict['report_features'][batch_idx, selected_item_idx]                 # [batch, sample]
        sample_angles = (torch.rand_like(selected_angles) * self.sample_radius * 0.1) + selected_angles
        cartesian_samples = torch.polar(sample_abs, sample_angles.to(sample_abs.dtype))
        cartesian_samples = torch.stack([cartesian_samples.real, cartesian_samples.imag], -1)

        return SwapSampleInformation(cartesian_samples, selected_item_idx)


class WhiteNoiseStripExampleSampleGenerator(ExampleSampleGenerator):
    """
    Generates single channel image of mostly mean -1 white noise
    
    From the center pointing to the desired feature dimension output, there is a mean 1 white noise
    """

    required_task_variable_keys = {'report_features', 'swap_probabilities'}

    def __init__(self, image_size: int, strip_pixel_width: int) -> None:
        raise Exception(header = 'need to include batch size dimension here!')
        self.sample_shape = [1, image_size, image_size]
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
            torch.Tensor: Image tensor of shape [batch_size, 1, image_size, image_size] 
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
        y_centered = - (y_coords - image_size // 2)
        
        # Calculate the angle of each pixel relative to center
        angles = torch.atan2(y_centered, x_centered)
        
        # Normalize angles to be between 0 and 2π
        angles = (angles + 2 * math.pi) % (2 * math.pi)
        
        # Normalize selected angles and reshape for broadcasting
        selected_item_ang = selected_item_ang * 0.0 + 1.0

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
        positive_where = torch.logical_and((angle_diff <= angular_width.unsqueeze(0)/2), radii > strip_pixel_width)
        images = torch.where(positive_where, 1.0, -1.0)

        return images.unsqueeze(1)

    def generate_sample_set(self, num_samples: int, variable_dict: Dict[str, _T]) -> SwapSampleInformation:
        selection_pmf = variable_dict['swap_probabilities']
        selected_item_idx = torch.tensor(np.random.choice(selection_pmf.shape[0], size = num_samples, replace = True, p = selection_pmf))
        selected_item_ang = torch.tensor(variable_dict['report_features'])[selected_item_idx]

        noise_offset = self.generate_strip_image(image_size=self.sample_shape[-1], strip_pixel_width=self.strip_pixel_width, selected_item_ang=selected_item_ang)
        
        white_noise_images = (noise_offset * 0.5) + (torch.randn_like(noise_offset) * 0.15)
        return SwapSampleInformation(white_noise_images, selected_item_idx)

    def generate_sample_diagnostics(self, sample_set: _T, variable_dict: Dict[str, _T]) -> Tuple[_T, Dict[str, _T]]:
        import pdb; pdb.set_trace(header = 'get pmf by taking closest to report_feature')
        return closest_rate, {'magnitude': magnitudes}

    def display_samples(self, sample_set: Union[SwapSampleInformation, _T], axes: Axes) -> None:
        if isinstance(sample_set, SwapSampleInformation):
            samples = sample_set.sample_set[0].permute(1, 2, 0).cpu().numpy()
        else:
            samples = sample_set[0].permute(1, 2, 0).cpu().numpy()
        extent=[-samples.shape[1]/2., samples.shape[1]/2., -samples.shape[0]/2., samples.shape[0]/2. ]
        # axes.imshow(samples, cmap = 'grey', vmin = -1.0, vmax = +1.0, extent = extent)
        axes.imshow(samples, cmap = 'grey', extent = extent)
    
    def display_early_x0_pred_timeseries(self, sample_set: _T, axes: Axes, cmap: ScalarMappable) -> None:
        T = sample_set.shape[1]
        chosen_sample_indices = [0, T // 4, T // 2, (3 * T // 4)]
        chosen_sample_extents = [
            (-sample_set.shape[-1], 0, 0, sample_set.shape[-1]), # top left
            (0, sample_set.shape[-1], 0, sample_set.shape[-1]), # top right
            (-sample_set.shape[-1], 0, -sample_set.shape[-1], 0), # bottom left
            (0, sample_set.shape[-1], -sample_set.shape[-1], 0), # bottom right
        ]
        for cs_idx, cs_ext in zip(chosen_sample_indices, chosen_sample_extents):
            # axes.imshow(sample_set[0,cs_idx,0], cmap = 'grey', vmin = -4.5, vmax = + 4.5, extent = cs_ext)
            axes.imshow(sample_set[0,cs_idx,0], cmap = 'grey', extent = cs_ext)
            axes.scatter([cs_ext[0] + 2], [cs_ext[-1] - 2], color = cmap.to_rgba(T - cs_idx), s = 20)
        axes.plot([-sample_set.shape[-1], sample_set.shape[-1]], [0., 0.], color = 'red')
        axes.plot([0., 0.], [-sample_set.shape[-1], sample_set.shape[-1]], color = 'red')
