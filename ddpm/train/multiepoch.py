"""
10.01.25
Specifically for usage with MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
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
from ddpm.utils.plotting import symmetrize_and_square_axis
from ddpm import tasks, model
from ddpm.tasks.main.multiepoch import MultiEpochDiffusionTask


import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer

from matplotlib import pyplot as plt


args = ConfigNamepace.from_yaml_path(sys.argv[1])


device = "cuda" if torch.cuda.is_available() else "cpu"


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


# Set up logging
magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=num_timesteps)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])
timer = LoopTimer(num_trials)
[training_print_path], save_base, _ = configure_logging_paths(
    save_base, log_suffixes=[f"train"], index_new=True
)
all_individual_residual_mses = np.zeros([num_trials, num_timesteps])
all_prep_state_losses = np.zeros([num_trials])
all_delay_activity_losses = np.zeros([num_trials])
args.write_to_yaml(os.path.join(save_base, "args.yaml"))


# Generate noising schedule
# sigma2x_schedule = (ultimate_sigma2 - starting_sigma2) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
# sigma2x_schedule = (starting_sigma2 + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_timesteps)
sigma2x_schedule = sigma2x_schedule.to(device=device)


task: MultiEpochDiffusionTask = getattr(tasks, task_name)(**task_config.dict)
task.save_metadata(os.path.join(save_base, "task_metadata"))


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


# Set up training
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

    total_loss = total_loss + (regularise_prep_state_weight * prep_state_loss) + (delay_activity_loss * regularise_prep_activity_weight)

    if total_loss.isnan() or total_loss.isinf():
        torch.save(prev_state, os.path.join(save_base, f"state_saved.mdl"))
        raise Exception(f'Total_loss goes to NaN or inf. {os.path.join(save_base, f"state_saved.mdl")} has latest safe state dict')

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    if t % 100_000 == 0:
        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.{t}.mdl"))
        torch.save(optim.state_dict(), os.path.join(save_base, f"opt_state.{t}.mdl"))

    if t >= plotting_start:
        all_individual_residual_mses[t - plotting_start, :] = (
            residual_mse.detach().cpu().mean(0).mean(0)
        )
        all_prep_state_losses[t - plotting_start] = prep_state_loss.detach().cpu()
        all_delay_activity_losses[t - plotting_start] = delay_activity_loss.detach().cpu()

    if (t - plotting_offset) % logging_freq == 0:

        test_trial_information = task.generate_trial_information(
            batch_size=1, num_samples=500
        )

        with torch.no_grad():
            test_forward_process = ddpm_model.noise(
                x_0=test_trial_information.sample_information.sample_set.to(device).float()
            )
            novel_samples_prep_dicts, novel_samples_dict = ddpm_model.generate_samples(
                prep_network_inputs=[
                    pni[[0]] for pni in test_trial_information.prep_network_inputs
                ],
                diffusion_network_inputs=[
                    dni[[0]] for dni in test_trial_information.diffusion_network_inputs
                ],
                prep_epoch_durations=test_trial_information.prep_epoch_durations,
                diffusion_epoch_durations=test_trial_information.diffusion_epoch_durations,
                samples_shape=[1, 500],
                noise_scaler=1.0,
            )

        fig, axes = plt.subplots(3, 5, figsize=(25, 15))

        axes[0, 0].set_title("Real sample(s)")
        axes[0, 1].set_title("Generated sample(s)")
        axes[0, 2].set_title("Early predictions of $x_0$")
        axes[0, 3].set_title(
            "Samples from base distribution vs actual pre-diffusion behaviour states"
        )
        axes[0, 4].set_title("$\hat\epsilon$ during generation")

        task.sample_gen.display_samples(
            test_trial_information.sample_information, axes[0, 0]
        )
        task.sample_gen.display_samples(novel_samples_dict["samples"], axes[0, 1])
        task.sample_gen.display_early_x0_pred_timeseries(
            novel_samples_dict["early_x0_preds"], axes[0, 2], kl_colors_scalarMap
        )
        task.sample_gen.display_samples(
            novel_samples_prep_dicts[-1]["postprep_base_samples"].detach().cpu(),
            axes[0, 3],
        )
        task.sample_gen.display_samples(
            test_forward_process["x_t"][:, :, -1, ...], axes[0, 3]
        )
        task.sample_gen.display_early_x0_pred_timeseries(
            novel_samples_dict["epsilon_hat"].detach().cpu(),
            axes[0, 4],
            kl_colors_scalarMap,
        )

        task.task_variable_gen.display_task_variables(
            test_trial_information.task_variable_information, axes[1, 0], axes[1, 1]
        )

        symmetrize_and_square_axis(axes[0, 0])
        symmetrize_and_square_axis(axes[0, 1])
        symmetrize_and_square_axis(axes[0, 2])
        symmetrize_and_square_axis(axes[0, 3])
        symmetrize_and_square_axis(axes[0, 4])

        axes[1, 2].set_title("Individual timesteps MSE")
        # axes[1,1].set_title('Individual timesteps MSE (zoomed in)')
        axes[1, 3].set_title("MSE averaged over timesteps")
        if t > plotting_start:
            zoomed_start = int((t + 1 - plotting_start) / 3)
            for h, trace in enumerate(
                all_individual_residual_mses[: t + 1 - plotting_start].T
            ):
                color = kl_colors_scalarMap.to_rgba(h + 1)
                axes[1, 2].plot(trace, color=color)
                axes[2, 2].plot(trace[zoomed_start:], color=color)
                # axes[1,1].plot(trace[-int(2 * (t+1-plotting_start) / 3):], color = color)
            axes[1, 3].plot(
                all_individual_residual_mses[: t + 1 - plotting_start].mean(-1)
            )
            axes[2, 3].plot(
                all_individual_residual_mses[zoomed_start : t + 1 - plotting_start].mean(-1)
            )
        cax = inset_axes(axes[1, 0], width="30%", height=1.0, loc=3)
        plt.colorbar(
            kl_colors_scalarMap,
            cax=cax,
            ticks=range(1, num_timesteps, 10),
            orientation="vertical",
        )

        axes[1, 4].set_title("Prep state behaviour activity regulariser")
        if t > plotting_start:
            axes[1, 4].plot(all_prep_state_losses[: t + 1 - plotting_start])

        axes[2, 4].set_title("Delay activity square mag")
        if t > plotting_start:
            axes[2, 4].plot(all_delay_activity_losses[: t + 1 - plotting_start])

        fig.savefig(os.path.join(save_base, "latest_log.png"))
        plt.close("all")

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.mdl"))
        torch.save(optim.state_dict(), os.path.join(save_base, f"opt_state.mdl"))

        if 'palimpsest' in task_name:

            fig, (stax, cax) = plt.subplots(1, 2, figsize = (10, 5))

            stax.imshow(test_trial_information.prep_network_inputs[0][0,0].cpu().reshape(task.sensory_gen.probe_num_tc, task.sensory_gen.report_num_tc))
            cax.imshow(test_trial_information.prep_network_inputs[2][0,0].cpu().reshape(task.sensory_gen.probe_num_tc, task.sensory_gen.report_num_tc))

            fig.savefig(os.path.join(save_base, "palimpsest_reprs.png"))


