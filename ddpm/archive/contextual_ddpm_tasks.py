raise Exception

import torch

import numpy as np
from torch import Tensor as _T

from purias_utils.multiitem_working_memory.util.circle_utils import (
    generate_circular_feature_list,
    polar2cart,
)

try:
    from sampling_ddpm.ddpm.model import DoublyConditionedDDPMReverseProcess
except ImportError:
    from ddpm.model import DoublyConditionedDDPMReverseProcess

from typing import Dict


def embed_2D_items_in_clifford(
    n_samples: int, torus_directions: Dict[str, _T], N_items: int = 3, radius=5.0
):
    """
    radius > 1.0 required because std = 1.0 for base distribution, i.e. p(x_T | M) = N(mu(M), I)
    """
    # Generate stimulus array M
    probe_features = generate_circular_feature_list(N_items, torch.pi / 4)
    report_features = generate_circular_feature_list(N_items, torch.pi / 4)
    probe_features_xy = torch.tensor(np.stack(polar2cart(1.0, probe_features), -1))
    report_features_xy = torch.tensor(np.stack(polar2cart(1.0, report_features), -1))

    # Toroidal representation (4D)
    probe_clifford = probe_features_xy.cuda().float().reshape(-1)
    report_clifford = report_features_xy.cuda().float().reshape(-1)

    # Embed the torus (N dimensional) then generate noisy base samples --> x_T
    embedded_mean = radius * (
        probe_clifford @ torus_directions["probe_directions"]
    ) + radius * (report_clifford @ torus_directions["report_directions"])
    repeated_embedded_mean = embedded_mean.unsqueeze(0).repeat(n_samples, 1)
    base_samples = repeated_embedded_mean + torch.randn_like(repeated_embedded_mean)

    # Evenly split the cued item between samples
    network_inputs = torch.zeros(n_samples, 2)
    network_inputs_idx = torch.zeros(n_samples)
    y_samples_means = torch.zeros(n_samples, 2)

    m = np.random.randint(0, N_items)
    network_inputs[:] = probe_features_xy[m]
    network_inputs_idx[:] = m
    y_samples_means[:] = report_features_xy[m]
    # for m in range(N_items):
    #     network_inputs[m::N_items] = probe_features_xy[m]
    #     network_inputs_idx[m::N_items] = m
    #     y_samples_means[m::N_items] = report_features_xy[m]

    y_samples = y_samples_means + 0.05 * torch.randn_like(y_samples_means)

    return {
        "base_samples": base_samples,
        "repeated_embedded_mean": repeated_embedded_mean,
        "y_samples": y_samples,
        "network_inputs": network_inputs,
        "network_inputs_idx": network_inputs_idx,
        "probe_features": probe_features,
        "report_features": report_features,
        "probe_clifford": probe_clifford,
        "report_clifford": report_clifford,
    }
