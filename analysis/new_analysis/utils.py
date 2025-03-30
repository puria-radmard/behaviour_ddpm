import torch
from torch import Tensor as _T

from typing import List

from matplotlib.pyplot import Axes

from analysis.decoders import AllemanStyleRoleReportFeatureProjector




def get_precue_elbo(probe_features: _T, report_features: _T, mode_generator: AllemanStyleRoleReportFeatureProjector, first_delay_activity: _T, p_correct: _T, do_elbo: bool):
    """
    report_features of shape [B, N, R]
    first_delay_activity of shape [B, T]
    p_correct: scalar or [B]
    """

    misbound_report_features = report_features[:,[1, 0],:]
    p_swap = 1.0 - p_correct
    
    nominal_modes = mode_generator.get_mixture_model_means_precue(report_features=report_features, probe_features=probe_features)            # [B, T, R]
    misbound_modes = mode_generator.get_mixture_model_means_precue(report_features=misbound_report_features, probe_features=probe_features)            # [B, T, R]

    variance = mode_generator.mode_variance_raw.exp()[None, None]   # [1, 1, R]
    nominal_distance_llh = mode_generator.get_gaussian_llh(first_delay_activity, nominal_modes, variance)         # [B, T]
    misbound_distance_llh = mode_generator.get_gaussian_llh(first_delay_activity, misbound_modes, variance)

    p_misbind = mode_generator.p_misbind

    nominal_repr_prior = p_correct + (1.0 - p_misbind) * p_swap
    misbound_repr_prior = p_misbind * p_swap

    if do_elbo:
        total_elbo = (
            (nominal_repr_prior * nominal_distance_llh) +
            (misbound_repr_prior * misbound_distance_llh) 
        ).mean(0)      # [T]
    else:
        total_elbo = (
            (nominal_repr_prior * nominal_distance_llh.exp()) +
            (misbound_repr_prior * misbound_distance_llh.exp()) 
        ).mean(0)      # [T]

    return total_elbo, nominal_modes, misbound_modes



def get_postcue_elbo(probe_features: _T, report_features: _T, mode_generator: AllemanStyleRoleReportFeatureProjector, diffusion_time_activity: _T, cued_indices: _T, p_correct: _T, do_elbo: bool):
    """
    report_features of shape [B, N, 2]
    diffusion_time_activity of shape [B, S (1), T, R]
    cued_indices: [B]
    p_correct: [B] or scalar
    """

    misselected_report_features = report_features[:,[1, 0],:]
    p_swap = 1.0 - p_correct

    nominal_modes = mode_generator.get_mixture_model_means_postcue(report_features=report_features, probe_features=probe_features, cued_indices=cued_indices)
    misselected_modes = mode_generator.get_mixture_model_means_postcue(report_features=misselected_report_features, probe_features=probe_features, cued_indices=cued_indices)
    misinterpreted_modes = mode_generator.get_mixture_model_means_postcue(report_features=report_features, probe_features=probe_features, cued_indices=1.0 - cued_indices)

    variance = mode_generator.mode_variance_raw.exp()[None, None]   # [1, 1, R]

    nominal_distance_llh = mode_generator.get_gaussian_llh(diffusion_time_activity, nominal_modes, variance)         # [B, T]
    misselected_distance_llh = mode_generator.get_gaussian_llh(diffusion_time_activity, misselected_modes, variance)         # [B, T]
    misinterpreted_distance_llh = mode_generator.get_gaussian_llh(diffusion_time_activity, misinterpreted_modes, variance)         # [B, T]

    p_goodcue, p_misselection, p_misinterpretation = mode_generator.p_post_cue_errors

    nominal_repr_prior = p_correct + p_goodcue * p_swap
    misselected_repr_prior = p_misselection * p_swap
    misinterpreted_repr_prior = p_misinterpretation * p_swap

    if do_elbo:
        total_elbo = (
            (nominal_repr_prior * nominal_distance_llh) +
            (misselected_repr_prior * misselected_distance_llh) +
            (misinterpreted_repr_prior * misinterpreted_distance_llh)
        ).mean(0)      # [T]
    else:
        total_elbo = (
            (nominal_repr_prior * nominal_distance_llh.exp()) +
            (misselected_repr_prior * misselected_distance_llh.exp()) +
            (misinterpreted_repr_prior * misinterpreted_distance_llh.exp())
        ).mean(0)      # [T]

    return total_elbo, nominal_modes, misselected_modes, misinterpreted_modes



@torch.no_grad()
def get_normalised_projection_with_moments(nominal_repr: _T, erroneous_repr: _T, activity: _T, correct_trials_mask: _T):
    """
    nominal_repr, erroneous_repr, activity: [B, T, R]
    correct_trials_mask <bool>: [B]
    """
    normal_to_error = erroneous_repr - nominal_repr
    nominal_to_activity = activity - nominal_repr

    normalised_projection = (normal_to_error * nominal_to_activity).sum(-1) / (normal_to_error * normal_to_error).sum(-1)

    correct_trials_normalised_projection = normalised_projection[correct_trials_mask]
    swap_trials_normalised_projection = normalised_projection[~correct_trials_mask]

    return correct_trials_normalised_projection, swap_trials_normalised_projection


def display_normalised_projection_timeseries(correct_trials_normalised_projection: _T, swap_trials_normalised_projection: _T, axes: List[Axes], swap_trials_color: str):

    correct_trials_mean_normalised_projection = correct_trials_normalised_projection.mean(0)
    correct_trials_std_normalised_projection = correct_trials_normalised_projection.std(0)
    correct_trials_upper_normalised_projection = correct_trials_mean_normalised_projection + 2 * correct_trials_std_normalised_projection
    correct_trials_lower_normalised_projection = correct_trials_mean_normalised_projection - 2 * correct_trials_std_normalised_projection
    
    swap_trials_mean_normalised_projection = swap_trials_normalised_projection.mean(0)
    swap_trials_std_normalised_projection = swap_trials_normalised_projection.std(0)
    swap_trials_upper_normalised_projection = swap_trials_mean_normalised_projection + 2 * swap_trials_std_normalised_projection
    swap_trials_lower_normalised_projection = swap_trials_mean_normalised_projection - 2 * swap_trials_std_normalised_projection

    duration = swap_trials_upper_normalised_projection.shape[0]

    axes[0].plot(correct_trials_mean_normalised_projection.cpu().numpy(), color = 'green')
    axes[0].plot(swap_trials_mean_normalised_projection.cpu().numpy(), color = swap_trials_color)

    axes[1].plot(correct_trials_mean_normalised_projection.cpu().numpy(), color = 'green')
    axes[1].fill_between(range(duration), correct_trials_lower_normalised_projection.cpu().numpy(), correct_trials_upper_normalised_projection.cpu().numpy(), color = 'green', alpha = 0.3)
    axes[1].plot(swap_trials_mean_normalised_projection.cpu().numpy(), color = swap_trials_color)
    axes[1].fill_between(range(duration), swap_trials_lower_normalised_projection.cpu().numpy(), swap_trials_upper_normalised_projection.cpu().numpy(), color = swap_trials_color, alpha = 0.3)

    axes[2].plot(correct_trials_mean_normalised_projection.cpu().numpy(), color = 'green')
    axes[2].fill_between(range(duration), correct_trials_lower_normalised_projection.cpu().numpy(), correct_trials_upper_normalised_projection.cpu().numpy(), color = 'green', alpha = 0.3)
    axes[2].plot(swap_trials_mean_normalised_projection.cpu().numpy(), color = swap_trials_color)
    axes[2].fill_between(range(duration), swap_trials_lower_normalised_projection.cpu().numpy(), swap_trials_upper_normalised_projection.cpu().numpy(), color = swap_trials_color, alpha = 0.3)

    ylim = axes[2].get_ylim()
    for ctnp in correct_trials_normalised_projection:
        axes[2].plot(ctnp.cpu().numpy(), color = 'green', alpha = 0.01)
    for stnp in swap_trials_normalised_projection:
        axes[2].plot(stnp.cpu().numpy(), color = swap_trials_color, alpha = 0.01)
    axes[2].set_ylim(ylim)
