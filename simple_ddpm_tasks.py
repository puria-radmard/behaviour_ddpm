import torch
import numpy as np


from typing import Dict, Tuple, Callable, Optional
from torch import Tensor as _T

from purias_utils.multiitem_working_memory.util.circle_utils import generate_circular_feature_list, rectify_angles, polar2cart



SPIRAL_MANIFOLD_MIN_THETA = 1
SPIRAL_MANIFOLD_MAX_THETA = 20
SPIRAL_MANIFOLD_RAD_GRAD = 0.2

MAX_INTRINSIC_COORD = 0.5 * SPIRAL_MANIFOLD_RAD_GRAD * (SPIRAL_MANIFOLD_MAX_THETA**2 - SPIRAL_MANIFOLD_MIN_THETA**2)
MIN_INTRINSIC_COORD = 0.5 * SPIRAL_MANIFOLD_RAD_GRAD * SPIRAL_MANIFOLD_MIN_THETA**2



TEST_INTRINSIC_COORDS = np.linspace(MIN_INTRINSIC_COORD, MAX_INTRINSIC_COORD, 1000)
SPIRAL_MANIFOLD_THETAS = (2 * TEST_INTRINSIC_COORDS / SPIRAL_MANIFOLD_RAD_GRAD) ** 0.5
SPIRAL_MANIFOLD_RS = SPIRAL_MANIFOLD_THETAS * SPIRAL_MANIFOLD_RAD_GRAD
SPIRAL_MANIFOLD_X, SPIRAL_MANIFOLD_Y = polar2cart(SPIRAL_MANIFOLD_RS, SPIRAL_MANIFOLD_THETAS)





def generate_uniform_spiral_manifolds(n_samples, include_inputs):
    if include_inputs:
        raise NotImplementedError
    equispaced_intrinsic_samples = np.random.uniform(MIN_INTRINSIC_COORD, MAX_INTRINSIC_COORD, n_samples)
    intrinsically_equispaced_thetas = (2 * equispaced_intrinsic_samples / SPIRAL_MANIFOLD_RAD_GRAD) ** 0.5    
    intrinsically_equispaced_rs = intrinsically_equispaced_thetas * SPIRAL_MANIFOLD_RAD_GRAD
    x, y = polar2cart(intrinsically_equispaced_rs, intrinsically_equispaced_thetas)
    return torch.tensor(
        np.stack([x, y], -1)
    ), None, None


def generate_random_gaussian_mixture(n_samples=2048, num_modes = 4):
    """
    Generate samples from a mixture of 4 2D Gaussians
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        samples: Tensor of shape (n_samples, 2) containing the samples

    Made, with edits, by Claude
    """

    assert 0 < num_modes <= 4
    
    means = 0.75 * torch.tensor([
        [ 3.4054,  2.9700],
        [-3.5293, -1.3912],
        [-2.9172,  1.9364],
        [ 2.8911, -0.3904],
    ])[:num_modes]
    
    # Generate different rotation angles for each Gaussian
    angles = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    # Different eigenvalues for different eccentricities
    eigenvals = torch.tensor([
        [0.2, 0.05],  # Very eccentric
        [0.5, 0.1],   # Highly eccentric
        [0.3, 0.2],   # Moderately eccentric
        [0.4, 0.4],   # Circular
    ])[:num_modes]
    
    # Initialize storage for samples
    samples = torch.zeros(n_samples, 2)
    
    # Equal weights for each Gaussian
    weights = torch.ones(num_modes) / 4
    
    # Generate samples
    n_per_gaussian = torch.multinomial(weights, n_samples, replacement=True)
    
    current_idx = 0
    for i in range(num_modes):
        n = n_per_gaussian.eq(i).sum().item()
        if n == 0:
            continue
            
        # Create rotation matrix
        angle = angles[i]
        rotation = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ])
        
        # Create covariance matrix = R * D * R^T
        cov = rotation @ torch.diag(eigenvals[i]) @ rotation.t()
        
        # Generate samples for this Gaussian
        dist = torch.distributions.MultivariateNormal(means[i], covariance_matrix=cov)
        samples[current_idx:current_idx + n] = dist.sample((n,))
        current_idx += n
        
    return samples




# CIRCLE_OFFSET = np.array([[-2, -2]])
# CIRCLE_RADIUS = 1.25
# SQUARE_OFFSET = np.array([[2, 2]])
CIRCLE_OFFSET = np.array([[-1., -1.]])
CIRCLE_RADIUS = 0.8
SQUARE_OFFSET = np.array([[1., 1.]])

SQUARE_SIDE_LENGTH = 2 * np.pi * CIRCLE_RADIUS / 4
SQUARE_BOUNDS = [0.0, 0.25, 0.50, 0.75, 1.0]
SQUARE_START_POINTS = np.array(
    [[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]
)[None] * 0.5 * SQUARE_SIDE_LENGTH + CIRCLE_OFFSET # Going anticlockwise form top right
SQUARE_END_POINTS = np.roll(SQUARE_START_POINTS, 1, 1)

SHAPES_WRONG_RATE = 0.2

def generate_little_shapes_from_intrinsic_coords(test_points):
    output = np.zeros([len(test_points), 2])
    on_circle_mask = test_points < 0.0                         # [N]
    circle_angles = test_points[on_circle_mask] * 2 * np.pi    # [N]   reverse order!
    circle_x, circle_y = polar2cart(CIRCLE_RADIUS, circle_angles)
    output[on_circle_mask, :] = np.stack([circle_x, circle_y], -1) + SQUARE_OFFSET

    for i, (lower, upper) in enumerate(zip(SQUARE_BOUNDS[:-1], SQUARE_BOUNDS[1:])):
        mask = np.logical_and(test_points >= lower, test_points < upper)
        frac = (test_points[mask][...,None] - lower) / (upper - lower)
        output[mask] = SQUARE_START_POINTS[:,i] * (1 - frac) + SQUARE_END_POINTS[:,i] * frac
    
    return output

def generate_little_shapes(n_samples, include_inputs):
    """
    Intrinsic coordinate goes around a circle, then around a square of the same perimeter => equal probability
    Below -1 to 0 = circle, 0 to +1 = square
    """
    if include_inputs:
        network_input_idxs = np.random.choice(2, n_samples, replace = True)      # {0, 1}^[N]
        equispaced_intrinsic_samples = (2*network_input_idxs.astype(float) - 1) * np.random.uniform(0, 1, n_samples)     # [N]
        network_inputs = np.zeros([n_samples, 2])
        network_inputs[range(n_samples), network_input_idxs.astype(int)] = 1.0
    else:
        equispaced_intrinsic_samples = np.random.uniform(-1, 1, n_samples)     # [N]
        network_input_idxs = None
        network_inputs = None
    return torch.tensor(generate_little_shapes_from_intrinsic_coords(equispaced_intrinsic_samples)), torch.tensor(network_inputs), torch.tensor(network_input_idxs)

def generate_little_shapes_uncertain(n_samples, include_inputs):
    assert include_inputs, "generate_little_shapes_uncertain task makes no sense without include_inputs flag!"
    network_input_idxs = np.random.choice(2, n_samples, replace = True)      # {0, 1}^[N]
    correct_choice = np.random.choice(2, n_samples, p=[1 - SHAPES_WRONG_RATE, SHAPES_WRONG_RATE], replace = True)     # {0, 1}^[N], 1 = correct shape chosen!
    equispaced_intrinsic_samples = (2*network_input_idxs.astype(float) - 1) * np.random.uniform(0, 1, n_samples)     # [N]
    equispaced_intrinsic_samples = (2*correct_choice.astype(float) - 1) * equispaced_intrinsic_samples
    network_inputs = np.zeros([n_samples, 2])
    network_inputs[range(n_samples), network_input_idxs.astype(int)] = 1.0
    network_input_idxs = network_input_idxs * 2 + (1 - correct_choice)   # (0,1) is first shape; (3,4) is second shape, both in order of (correct, swap)
    return torch.tensor(generate_little_shapes_from_intrinsic_coords(equispaced_intrinsic_samples)), torch.tensor(network_inputs), torch.tensor(network_input_idxs)

SHAPES_MANIFOLD_X, SHAPES_MANIFOLD_Y = generate_little_shapes_from_intrinsic_coords(np.linspace(-1, 1, 1000, endpoint=False)).T



MULTIITEM_MIN_MARGIN = torch.pi / 6
# MULTIITEM_N_ITEMS = 4
# MULTIITEM_CATEGORICAL_WEIGHTS = torch.tensor([0.7, 0.15, 0.075, 0.075])
MULTIITEM_N_ITEMS = 2
MULTIITEM_CATEGORICAL_WEIGHTS = torch.tensor([0.7, 0.3])
MULTIITEM_CATEGORICAL = torch.distributions.Categorical(MULTIITEM_CATEGORICAL_WEIGHTS)
MULTIITEM_EMISSIONS_STD = 0.05


def generate_single_item(n_samples, include_inputs):
    assert include_inputs, "generate_single_item task makes no sense without include_inputs flag!"
    features = generate_circular_feature_list(1, MULTIITEM_MIN_MARGIN)
    repeated_features = rectify_angles(torch.tensor(features).repeat(n_samples, 1))
    selected_features = repeated_features.squeeze(-1)
    x, y = polar2cart(1.0, selected_features)
    selected_xy_coords = torch.tensor(np.stack([x, y], -1))
    selected_xy_coords = selected_xy_coords + torch.randn_like(selected_xy_coords) * MULTIITEM_EMISSIONS_STD
    return selected_xy_coords, repeated_features, torch.zeros(n_samples)

def generate_multiitem(n_samples, include_inputs):
    assert include_inputs, "generate_multiitem task makes no sense without include_inputs flag!"
    features = generate_circular_feature_list(MULTIITEM_N_ITEMS, MULTIITEM_MIN_MARGIN)
    selected_idx = MULTIITEM_CATEGORICAL.sample([n_samples])
    repeated_features = rectify_angles(torch.tensor(features).repeat(n_samples, 1))
    selected_features = torch.zeros(n_samples, MULTIITEM_N_ITEMS).to(repeated_features.dtype)
    selected_features[range(n_samples),selected_idx] = repeated_features[range(n_samples),selected_idx]
    selected_features = selected_features.sum(-1)
    x, y = polar2cart(1.0, selected_features)
    selected_xy_coords = torch.tensor(np.stack([x, y], -1))
    selected_xy_coords = selected_xy_coords + torch.randn_like(selected_xy_coords) * MULTIITEM_EMISSIONS_STD
    input_x, input_y = polar2cart(1.0, repeated_features)
    input_features = torch.concat([input_x, input_y], -1)
    return selected_xy_coords, input_features, selected_idx



UNIT_CIRCLE_X, UNIT_CIRCLE_Y = polar2cart(1.0, torch.linspace(0.0, 2*torch.pi, 100))






_task_function_type = Callable[[int, bool], Tuple[_T, Optional[_T], Optional[_T]]]

MANIFOLD_INFORMATION_DICTIONARY: Dict[str, Tuple[_task_function_type, _T, _T, int]] = {
    'spiral': (generate_uniform_spiral_manifolds, SPIRAL_MANIFOLD_X, SPIRAL_MANIFOLD_Y, 2, False),
    'shapes': (generate_little_shapes, SHAPES_MANIFOLD_X, SHAPES_MANIFOLD_Y, 2, False),
    'shapes_uncertain': (generate_little_shapes_uncertain, SHAPES_MANIFOLD_X, SHAPES_MANIFOLD_Y, 2, False),
    'simple_multiitem': (generate_multiitem, UNIT_CIRCLE_X, UNIT_CIRCLE_Y, MULTIITEM_N_ITEMS * 2, False),
    'simple_singleitem': (generate_single_item, UNIT_CIRCLE_X, UNIT_CIRCLE_Y, 1, True),
}


if __name__ == '__main__':
    # Generate samples
    samples = generate_little_shapes(n_samples=1000)
    print(samples.shape)

    # Optional: visualize the samples
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, color='gray')
    plt.axis('equal')
    plt.title('Samples from Shapes Manifolds')
    plt.savefig('swirls')




