"""
10.01.25
Specifically for usage with MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
Modified to include hazard rate loss
"""

import os
import sys
import copy
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from ddpm.model import BouncePopulationResidualModel

from purias_utils.util.arguments_yaml import ConfigNamepace

from ddpm.model.main.multiepoch import (
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
)
from ddpm import tasks, model
from ddpm.tasks.main.multiepoch import MultiEpochDiffusionTask
from ddpm.utils.vis import plot_standard_losses_multiepoch, imshow_palimpsest_reprs, scatter_standard_responses_multiepoch, plot_mse_losses

# Import hazard rate utilities
from ddpm.utils.hazard_rate import (
    log_normal_to_discrete,
    response_time_loss,
    sample_response_times_batch_trialwise,
)

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer

from matplotlib import pyplot as plt


args = ConfigNamepace.from_yaml_path(sys.argv[1])


device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_swap_dependent_lognormal_params(batch_size, num_trials, was_swapped, device):
    """
    Generate log-normal parameters that depend on swap status.
    
    Args:
        batch_size: int
        num_trials: int
        was_swapped: [batch_size, num_trials] - boolean array indicating swap status
        device: torch device
        
    Returns:
        log_means: [batch_size, num_trials] - μ parameters
        log_vars: [batch_size, num_trials] - σ² parameters
    """
    # Short RT for swapped trials: μ = -0.8, σ² = 0.04 
    # (median ≈ 0.45s ≈ 9 timesteps, 99.9th percentile ≈ 1.0s ≈ 20 timesteps)
    short_mean = -0.8
    short_var = 0.04
    
    # Long RT for non-swapped trials: μ = -0.3, σ² = 0.16
    # (median ≈ 0.74s ≈ 15 timesteps, 99.9th percentile ≈ 2.0s ≈ 40 timesteps)  
    long_mean = -0.3
    long_var = 0.16
    
    # Initialize arrays
    log_means = torch.zeros(batch_size, num_trials, device=device)
    log_vars = torch.zeros(batch_size, num_trials, device=device)
    
    # Fill based on swap status
    log_means[was_swapped] = short_mean
    log_means[~was_swapped] = long_mean
    log_vars[was_swapped] = short_var
    log_vars[~was_swapped] = long_var
    
    return log_means, log_vars


# Unpack all args
ultimate_sigma2 = args.ultimate_sigma2
starting_sigma2 = args.starting_sigma2
num_timesteps = args.num_timesteps
num_samples = args.num_samples
batch_size = args.batch_size
num_trials = args.num_trials
logging_freq = args.logging_freq
save_base = args.save_base
task_name = args.task_name
task_config = args.task_config
regularise_prep_state_weight = args.regularise_prep_state_weight
regularise_prep_activity_indices = args.regularise_prep_activity_indices
regularise_prep_activity_weight = args.regularise_prep_activity_weight
regularise_diffusion_nullspace = args.regularise_diffusion_nullspace
model_name = args.model_name
model_config = args.model_config
lr = args.lr
resume_path = args.resume_path

# New hazard rate arguments
reaction_time_loss_weight = getattr(args, 'reaction_time_loss_weight', 0.0)


# Set up logging
magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=num_timesteps)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])
timer = LoopTimer(num_trials)
[training_print_path], save_base, _ = configure_logging_paths(
    save_base, log_suffixes=[f"train"], index_new=True
)
all_prep_state_losses = np.zeros([num_trials])
all_delay_activity_losses = np.zeros([num_trials])
all_reaction_time_losses = np.zeros([num_trials])  # Track reaction time losses
args.write_to_yaml(os.path.join(save_base, "args.yaml"))


# Generate noising schedule
# sigma2x_schedule = (ultimate_sigma2 - starting_sigma2) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
# sigma2x_schedule = (starting_sigma2 + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_timesteps)
sigma2x_schedule = sigma2x_schedule.to(device=device)


task: MultiEpochDiffusionTask = getattr(tasks, task_name)(**task_config.dict)
task.save_metadata(os.path.join(save_base, "task_metadata"))
try:
    all_individual_residual_mses = np.zeros([num_trials, task.sample_gen.limit_training_timesteps])
    plotting_num_timesteps = task.sample_gen.limit_training_timesteps
except:
    all_individual_residual_mses = np.zeros([num_trials, num_timesteps])
    plotting_num_timesteps = num_timesteps

all_trial_type_trial_indices = {trial_type: [] for trial_type in task.task_variable_gen.trial_types}
all_trial_type_trial_indices['all'] = []
if len(task.task_variable_gen.trial_types) == 1:
    assert task.task_variable_gen.trial_types[0] == 'all'


# Set up model
residual_model_kwargs = model_config.dict.pop("residual_model_kwargs").dict
ddpm_model_kwargs = model_config.dict.pop("ddpm_model_kwargs").dict
ddpm_model, mse_key_pred, mse_key_target = getattr(model, model_name)(
    **model_config.dict,
    residual_model_kwargs=residual_model_kwargs,
    ddpm_model_kwargs=ddpm_model_kwargs,
    sigma2x_schedule=sigma2x_schedule,
    prep_sensory_shape=task.sensory_gen.prep_sensory_shape,
    underlying_sensory_shape=task.sensory_gen.underlying_sensory_shape,
    sample_shape=task.sample_gen.sample_shape,
    device=device,
)

ddpm_model: MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
ddpm_model.to(device)
mse_key_pred: str
# assert mse_key_pred == "epsilon_hat"


# Initialize hazard rate readout parameter
if reaction_time_loss_weight > 0:
    # W_hazard maps from behaviour nullspace to scalar hazard rate
    behaviour_nullspace_dim = ddpm_model.behaviour_nullspace.shape[0]  # Should be (num_features - 2)
    W_hazard = torch.nn.Parameter(torch.randn(behaviour_nullspace_dim, 1, device=device) * 0.01)
    optim = torch.optim.Adam(list(ddpm_model.parameters()) + [W_hazard], lr=lr)
else:
    W_hazard = None
    optim = torch.optim.Adam(ddpm_model.parameters(), lr=lr)


if resume_path is not None:
    task.load_metadata(resume_path.replace('state.mdl', 'task_metadata.npy'))
    trained_state_dict = torch.load(resume_path, weights_only=True)
    if args.resume_kept_input_dims is not None:
        ddpm_model.load_state_dict(trained_state_dict, kept_input_dims=args.resume_kept_input_dims)
    else:
        try:
            ddpm_model.load_state_dict(trained_state_dict)
        except AssertionError as e:
            print(e)
            print('warning: loading state_dict non-strictly')
            ddpm_model.load_state_dict(trained_state_dict, strict = False)
            if isinstance(ddpm_model.residual_model, BouncePopulationResidualModel):
                residual_model_state_dict = {k.removeprefix('residual_model.'): v for k, v in trained_state_dict.items() if k.startswith('residual_model.')}
                ddpm_model.residual_model.load_state_dict(state_dict=residual_model_state_dict)
                print('loaded BouncePopulationResidualModel from existing non bounce!')

    # Load W_hazard if it exists
    if W_hazard is not None:
        try:
            W_hazard_path = resume_path.replace('state.mdl', 'W_hazard.pt')
            if os.path.exists(W_hazard_path):
                W_hazard.data = torch.load(W_hazard_path, weights_only=True)
                print('Loaded W_hazard from checkpoint')
        except Exception as e:
            print(f'Could not load W_hazard: {e}')

    try:
        optim.load_state_dict(torch.load(resume_path.replace('state.mdl', 'opt_state.mdl'), weights_only=True))
    except Exception as e:
        print(e)


# For transparency
sch_fig, sch_axes = plt.subplots(2, 1, figsize=(5, 8))
sch_axes[0].plot(
    ddpm_model.sigma2xt_schedule.cpu().numpy(), label="sigma2xt_schedule", alpha=0.4
)
sch_axes[0].plot(ddpm_model.a_t_schedule.cpu().numpy(), label="a_t_schedule", alpha=0.4)
sch_axes[0].plot(
    ddpm_model.root_b_t_schedule.cpu().numpy(), label="root_b_t_schedule", alpha=0.4
)
sch_axes[0].plot(
    ddpm_model.noise_scaler_schedule.cpu().numpy(),
    label="noise_scaler_schedule",
    alpha=0.4,
)
sch_axes[0].plot(
    ddpm_model.base_samples_scaler_schedule.cpu().numpy(),
    label="base_samples_scaler_schedule",
    alpha=0.4,
)
sch_axes[0].plot(
    ddpm_model.residual_scaler_schedule.cpu().numpy(),
    label="residual_scaler_schedule",
    alpha=0.4,
)
sch_axes[0].legend()

sch_axes[1].set_title("Time embeddings")
sch_axes[1].imshow(ddpm_model.time_embeddings.time_embs.detach().cpu().numpy().T)

plt.savefig(os.path.join(save_base, "sigma_schedule_unrolling.png"))


# This will get filled in and continuously updated by task.sample_gen.generate_sample_diagnostics
recent_sample_diagnostics = deque(maxlen=100)

plotting_offset = 0
plotting_start = 0


for t in tqdm(range(num_trials)):

    prev_state = copy.deepcopy(ddpm_model.state_dict())

    timer.loop_start()

    trial_information = task.generate_trial_information(
        batch_size=batch_size, num_samples=num_samples
    )

    # Get actual swap status from trial information
    swapped_training_examples = (trial_information.sample_information.item_indices != trial_information.task_variable_information['cued_item_idx'][:,None])

    with torch.no_grad():
        forward_process = ddpm_model.noise(
            x_0=trial_information.sample_information.sample_set.to(device).float()
        )
    preparatory_state_dicts, epsilon_hat_dict = ddpm_model.residual(
        x_samples=forward_process["x_t"],
        prep_network_inputs=trial_information.prep_network_inputs,
        diffusion_network_inputs=trial_information.diffusion_network_inputs,
        prep_epoch_durations=trial_information.prep_epoch_durations,
        diffusion_epoch_durations=trial_information.diffusion_epoch_durations,
    )
    residual_mse = task.sample_gen.mse(
        epsilon_hat_dict[mse_key_pred], forward_process[mse_key_target]
    )  # [batch, samples, time]

    total_loss = residual_mse.mean()

    prep_state_loss = (
        preparatory_state_dicts[-1]["postprep_base_samples"]
        .mean(-2)
        .square()
        .sum(-1)
        .sqrt()
        .mean(0)
    )  # [B,S,2] -> mean over samples [B,2] -> mag of mean [B] -> average of that <scalar>


    prep_activities_to_regularise = [preparatory_state_dicts[rpai]['preparatory_trajectory'] for rpai in regularise_prep_activity_indices]
    if regularise_diffusion_nullspace:
        prep_activities_to_regularise.append(epsilon_hat_dict["trajectories"] @ ddpm_model.behaviour_nullspace.T)

    delay_activity_loss = 0.0
    for patr in prep_activities_to_regularise:
        delay_activity_loss = delay_activity_loss + patr.square().sum(-1).sqrt().mean()  # [B, S, T, N] -> mags [B, S, T] -> average of that <scalar>

    # Compute reaction time loss
    reaction_time_loss = 0.0
    if reaction_time_loss_weight > 0 and W_hazard is not None:
        # Get trajectories: [batch, samples, time, features]
        trajectories = epsilon_hat_dict["trajectories"]
        batch_size_actual, num_samples_actual, time_steps, features = trajectories.shape
        
        # Project to behaviour nullspace, then to hazard rate
        # trajectories @ behaviour_nullspace.T @ W_hazard
        nullspace_trajectories = trajectories @ ddpm_model.behaviour_nullspace.T  # [B, S, T, nullspace_dim]
        hazard_logits = nullspace_trajectories @ W_hazard  # [B, S, T, 1]
        hazard_rates = torch.sigmoid(hazard_logits.squeeze(-1))  # [B, S, T]
        
        # Generate swap-dependent log-normal targets using actual swap status
        target_log_means, target_log_vars = generate_swap_dependent_lognormal_params(
            batch_size_actual, num_samples_actual, swapped_training_examples, device
        )

        # Compute reaction time loss (trial-wise, then average)
        rt_losses = response_time_loss(hazard_rates, target_log_means, target_log_vars, dt=0.05)  # [B, S]
        reaction_time_loss = rt_losses.mean()

    total_loss = total_loss + (regularise_prep_state_weight * prep_state_loss) + (delay_activity_loss * regularise_prep_activity_weight) + (reaction_time_loss_weight * reaction_time_loss)

    if total_loss.isnan() or total_loss.isinf():
        torch.save(prev_state, os.path.join(save_base, f"state_saved.mdl"))
        if W_hazard is not None:
            torch.save(W_hazard.data, os.path.join(save_base, f"W_hazard_saved.pt"))
        raise Exception(f'Total_loss goes to NaN or inf. {os.path.join(save_base, f"state_saved.mdl")} has latest safe state dict')

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    if 'trial_type_idx' in trial_information.task_variable_information:
        import pdb; pdb.set_trace()
    else:
        all_trial_type_trial_indices['all'].append(t)

    if t % 100_000 == 0:
        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.{t}.mdl"))
        torch.save(optim.state_dict(), os.path.join(save_base, f"opt_state.{t}.mdl"))
        if W_hazard is not None:
            torch.save(W_hazard.data, os.path.join(save_base, f"W_hazard.{t}.pt"))

    if t >= plotting_start:

        if len(all_trial_type_trial_indices) == 1:
            assert 'trial_type_idx' not in trial_information.task_variable_information
        all_individual_residual_mses[t - plotting_start, :] = (
            residual_mse.detach().cpu().mean(0).mean(0)
        )

        all_prep_state_losses[t - plotting_start] = prep_state_loss.detach().cpu()
        all_delay_activity_losses[t - plotting_start] = delay_activity_loss.detach().cpu()
        all_reaction_time_losses[t - plotting_start] = reaction_time_loss.detach().cpu() if isinstance(reaction_time_loss, torch.Tensor) else reaction_time_loss

    if (t - plotting_offset) % logging_freq == 0:

        test_trial_information = task.generate_test_trial_information(num_samples=1024)
        
        # fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig, axes = plt.subplots(1 + 2 * len(task.task_variable_gen.trial_types), 7, figsize=(35, 5 * (1 + 2 * len(task.task_variable_gen.trial_types))))

        plot_standard_losses_multiepoch(
            mse_ax = axes[0, 0], mean_mse_ax = axes[0, 1], 
            zoomed_mse_ax = axes[0, 2], zoomed_mean_mse_ax = axes[0, 3], 
            prep_state_reg_axes = axes[0, 4], delay_activity_reg_axes = axes[0, 5],
            training_step = t, plotting_start = plotting_start,
            diffusion_cmap = kl_colors_scalarMap,
            num_timesteps = plotting_num_timesteps, 
            all_individual_residual_mses = all_individual_residual_mses[all_trial_type_trial_indices['all']], 
            all_prep_state_losses = all_prep_state_losses, 
            all_delay_activity_losses = all_delay_activity_losses,
            trial_type_name = 'all'
        )

        # Plot reaction time loss if enabled
        if reaction_time_loss_weight > 0:
            axes[0, 6].plot(all_reaction_time_losses[:t-plotting_start+1])
            axes[0, 6].set_title('Reaction Time Loss')
            axes[0, 6].set_xlabel('Training Step')
            axes[0, 6].set_ylabel('RT Loss')
            axes[0, 6].grid(True, alpha=0.3)

        with torch.no_grad():
            test_forward_process = ddpm_model.noise(
                x_0=test_trial_information.sample_information.sample_set.to(device).float()
            )
            novel_samples_prep_dicts, novel_samples_dict = ddpm_model.generate_samples(
                prep_network_inputs=test_trial_information.prep_network_inputs,
                diffusion_network_inputs=test_trial_information.diffusion_network_inputs,
                prep_epoch_durations=test_trial_information.prep_epoch_durations,
                diffusion_epoch_durations=test_trial_information.diffusion_epoch_durations,
                samples_shape=[len(task.task_variable_gen.trial_types), 1024],
                noise_scaler=1.0,
            )

        for trial_type_idx, test_trial_type in enumerate(task.task_variable_gen.trial_types):

            trial_type_top_row_idx = trial_type_idx * 2 + 1

            scatter_standard_responses_multiepoch(
                real_samples_axes = axes[trial_type_top_row_idx, 0],
                generated_samples_axes = axes[trial_type_top_row_idx, 1],
                early_predictions_axes = axes[trial_type_top_row_idx, 2],
                start_of_prep_axes = axes[trial_type_top_row_idx, 3],
                pred_residual_axes = axes[trial_type_top_row_idx, 4],
                task = task,
                task_variable_axes = [axes[trial_type_top_row_idx + 1,0], axes[trial_type_top_row_idx + 1,1]],
                novel_samples_dict = novel_samples_dict,
                novel_samples_prep_dicts = novel_samples_prep_dicts,
                forward_process_dict = test_forward_process,
                trial_info = test_trial_information,
                diffusion_cmap = kl_colors_scalarMap,
                trial_type_name = test_trial_type,
                batch_idx = trial_type_idx
            )

            plot_mse_losses(
                mse_ax = axes[trial_type_top_row_idx, 5], 
                mean_mse_ax = axes[trial_type_top_row_idx, 6], 
                zoomed_mse_ax = axes[trial_type_top_row_idx+1, 5],  
                zoomed_mean_mse_ax = axes[trial_type_top_row_idx+1, 6], 
                training_step = t, 
                plotting_start = plotting_start, 
                num_timesteps = plotting_num_timesteps, 
                diffusion_cmap = kl_colors_scalarMap, 
                trial_type_name = test_trial_type,
                all_individual_residual_mses = all_individual_residual_mses[all_trial_type_trial_indices[test_trial_type]]
            )

            if 'palimpsest' in task_name:
                stax, cax = [axes[trial_type_top_row_idx + 1,2], axes[trial_type_top_row_idx + 1,3]]
                imshow_palimpsest_reprs([stax, cax], test_trial_information, task, [0, 2], trial_type_idx)
        
        fig.savefig(os.path.join(save_base, "latest_log.png"))
        plt.close(fig)

        # Generate hazard rate diagnostics if enabled
        if reaction_time_loss_weight > 0 and W_hazard is not None:

            assert args.task_config.sample_size == 2
            assert len(task.task_variable_gen.trial_types) == 1

            with torch.no_grad():
                # Classify novel trials into swapped and not swapped - result = [num trials]
                all_novel_errors = (novel_samples_dict['samples'].squeeze(0)[:,None] - args.task_config.sample_radius * test_trial_information.task_variable_information['report_features_cart'])    # [trials, items, 2]
                all_novel_errors = all_novel_errors.square().sum(-1).sqrt()
                actual_errors = all_novel_errors[:, test_trial_information.task_variable_information['cued_item_idx'].item()]
                swapped_errors = all_novel_errors[:, 1 - test_trial_information.task_variable_information['cued_item_idx'].item()]
                swapped_trials = torch.abs(actual_errors) > torch.abs(swapped_errors)

                # Sample novel finishing times
                novel_hazard_rates = torch.sigmoid(novel_samples_dict['embedded_sample_trajectory'] @ ddpm_model.behaviour_nullspace.T.cpu() @ W_hazard.cpu()).squeeze(-1)    # [test batch (1), num trials, num timesteps]
                train_rt_samples = sample_response_times_batch_trialwise(novel_hazard_rates, samples_per_trial=1).squeeze(-1).squeeze(0)           # [num trials]

            train_time_steps = novel_hazard_rates.shape[-1]

            # For histograms, create separate examples for swapped vs non-swapped
            # Generate targets for histogram examples
            # import pdb; pdb.set_trace(header = "replace with target_log_means, target_log_vars??")
            swap_target_means, swap_target_vars = generate_swap_dependent_lognormal_params(
                1, 1, torch.tensor([[True]], device=device), device
            )
            noswap_target_means, noswap_target_vars = generate_swap_dependent_lognormal_params(
                1, 1, torch.tensor([[False]], device=device), device
            )
            
            # Generate target distributions for histograms
            swap_target_dist = log_normal_to_discrete(
                swap_target_means, swap_target_vars, dt=0.05, max_timesteps=train_time_steps
            ).cpu().numpy()[0, 0]
            
            noswap_target_dist = log_normal_to_discrete(
                noswap_target_means, noswap_target_vars, dt=0.05, max_timesteps=train_time_steps  
            ).cpu().numpy()[0, 0]

            
            # Seperate them into swapped and not swapped
            train_rt_samples_swapped = train_rt_samples[swapped_trials]
            train_rt_samples_not_swapped = train_rt_samples[~swapped_trials]

            # Create separate figure for hazard rate diagnostics
            hazard_fig, hazard_axes = plt.subplots(1, 1, figsize=(24, 18))

            time_bins = np.arange(train_time_steps + 1) - 0.5
            time_centers = np.arange(train_time_steps)

            hazard_axes.hist(train_rt_samples_swapped, bins=time_bins, density=True, alpha=0.7, 
                    label='Sampled RTs', color='lightcoral')
            hazard_axes.plot(time_centers, swap_target_dist, 'r-', linewidth=2, 
                    label='Swapped Target')

            hazard_axes.hist(train_rt_samples_not_swapped, bins=time_bins, density=True, alpha=0.7,
                    label='Sampled RTs', color='lightblue')
            hazard_axes.plot(time_centers, noswap_target_dist, 'b-', linewidth=2,
                    label='Non-Swapped Target')


            hazard_fig.savefig(os.path.join(save_base, "hazard_rate_diagnostics.png"), dpi=100)
            plt.close(hazard_fig)

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.mdl"))
        torch.save(optim.state_dict(), os.path.join(save_base, f"opt_state.mdl"))
        if W_hazard is not None:
            torch.save(W_hazard.data, os.path.join(save_base, f"W_hazard.pt"))
