import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from purias_utils.util.arguments_yaml import ConfigNamepace


from ddpm.model.main.scorematching import ScoreMatchingLinearSubspaceTeacherForcedDDPMReverseProcess
from ddpm.tasks.main.scorematching import ScoreMatchingDiffusionTask
from ddpm.utils.plotting import symmetrize_and_square_axis
from ddpm import tasks, model

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
state_space_size = args.state_space_size
recurrence_hidden_layers = args.recurrence_hidden_layers
time_embedding_size = args.time_embedding_size
ultimate_sigma2 = args.ultimate_sigma2
starting_sigma2 = args.starting_sigma2
num_timesteps = args.num_timesteps
noise_schedule_power = args.noise_schedule_power
num_samples = args.num_samples
batch_size = args.batch_size
num_trials = args.num_trials
logging_freq = args.logging_freq
save_base = args.save_base
task_name = args.task_name
task_config = args.task_config
regularise_prep_state = args.regularise_prep_state
if regularise_prep_state:
    regularise_prep_state_weight = args.regularise_prep_state_weight
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
if regularise_prep_state:
    all_prep_state_losses = np.zeros([num_trials])
args.write_to_yaml(os.path.join(save_base, "args.yaml"))


# Generate noising schedule
# sigma2x_schedule = (ultimate_sigma2 - starting_sigma2) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
# sigma2x_schedule = (starting_sigma2 + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_timesteps)
sigma2x_schedule = sigma2x_schedule.to(device=device)


# Set up task
task: ScoreMatchingDiffusionTask = getattr(tasks, task_name)(**task_config.dict)
task.save_metadata(os.path.join(save_base, "task_metadata"))


# Set up model
residual_model_kwargs = model_config.dict.pop("residual_model_kwargs").dict
ddpm_model_kwargs = model_config.dict.pop("ddpm_model_kwargs").dict
ddpm_model, mse_key = getattr(model, model_name)(
    **model_config.dict,
    residual_model_kwargs=residual_model_kwargs,
    ddpm_model_kwargs=ddpm_model_kwargs,
    sigma2x_schedule=sigma2x_schedule,
    sensory_shape=task.sensory_gen.sensory_shape,
    sample_shape=task.distribution_gen.sample_shape,
    device=device,
)

ddpm_model: ScoreMatchingLinearSubspaceTeacherForcedDDPMReverseProcess
ddpm_model.to(device)
mse_key: str

if resume_path is not None:
    ddpm_model.load_state_dict(torch.load(resume_path))


# Set up training
optim = torch.optim.Adam(ddpm_model.parameters(), lr=lr)


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

    timer.loop_start()

    trial_information = task.generate_trial_information(
        batch_size=batch_size, num_samples=num_samples
    )

    epsilon_hat_dict = ddpm_model.est_score(
        target_distribution=trial_information.distribution_information,
        network_input=trial_information.network_inputs,
    )

    true_score = trial_information.distribution_information.calculate_score(
        x_t = epsilon_hat_dict['subspace_trajectories'].flip(-2), # Reverse order!
        a_t = ddpm_model.a_t_schedule,
        b_t = ddpm_model.root_b_t_schedule.square(),
    )

    residual_mse = task.distribution_gen.score_mse(
        epsilon_hat_dict[mse_key],
        true_score
    )  # [batch, samples, time]

    total_loss = residual_mse.mean()

    if regularise_prep_state:
        prep_state_loss = (
            epsilon_hat_dict["postprep_base_samples"]
            .mean(-2)
            .square()
            .sum(-1)
            .sqrt()
            .mean(0)
        )  # [B,S,2] -> mean over samples [B,2] -> mag of mean [B] -> average of that <scalar>
        total_loss = total_loss + (regularise_prep_state_weight * prep_state_loss)

    optim.zero_grad()
    total_loss.backward()
    optim.step()


    if t >= plotting_start:
        all_individual_residual_mses[t - plotting_start, :] = (
            residual_mse.detach().cpu().mean(0).mean(0)
        )

    if (t - plotting_offset) % logging_freq == 0:

        with torch.no_grad():
            novel_samples_dict = ddpm_model.generate_samples(
                # trial_information.distribution_information,   # For debugging only
                network_input=trial_information.network_inputs[[0], :],
                samples_shape=[1, num_samples],
                noise_scaler=1.0,
            )

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))

        axes[0, 0].set_title("Real score")
        axes[0, 1].set_title("Generated sample(s) - no teacher forcing")
        axes[0, 2].set_title("Generated sample(s) - full teacher forcing")
        axes[0, 3].set_title(
            "Samples from pre-diffusion (preparatory) behaviour states"
        )
        axes[0, 4].set_title("$\hat s$ during generation")

        trial_information.distribution_information.display_final_score(axes = axes[0, 0])
        task.distribution_gen.display_samples(novel_samples_dict["samples"], axes[0, 1])
        task.distribution_gen.display_samples(epsilon_hat_dict["subspace_trajectories"][...,-1,:].detach(), axes[0, 2])
        trial_information.distribution_information.display_final_score(axes = axes[0, 1], plotting_kwargs={'alpha': 0.2}, use_axes_lims = True)
        trial_information.distribution_information.display_final_score(axes = axes[0, 2], plotting_kwargs={'alpha': 0.2}, use_axes_lims = True)

        task.distribution_gen.display_sample_timeseries(
            epsilon_hat_dict["score_hat"].detach().cpu(),
            axes[0, 4],
            cmap = kl_colors_scalarMap,
        )

        task.task_variable_gen.display_task_variables(
            trial_information.task_variable_information, axes[1, 0], axes[1, 1]
        )

        symmetrize_and_square_axis(axes[0, 0])
        symmetrize_and_square_axis(axes[0, 1])
        symmetrize_and_square_axis(axes[0, 2])
        symmetrize_and_square_axis(axes[0, 3])
        symmetrize_and_square_axis(axes[0, 4])

        axes[1, 2].set_title("Individual timesteps score MSE")
        # axes[1,1].set_title('Individual timesteps MSE (zoomed in)')
        axes[1, 3].set_title("Score MSE averaged over timesteps (zoomed in)")
        if t > plotting_start:
            end = t + 1 - plotting_start
            for h, trace in enumerate(
                all_individual_residual_mses[: end].T
            ):
                color = kl_colors_scalarMap.to_rgba(h + 1)
                axes[1, 2].plot(trace, color=color)
                # axes[1,1].plot(trace[-int(2 * (t+1-plotting_start) / 3):], color = color)
            start = int(end / 3)
            axes[1, 3].plot(
                all_individual_residual_mses[start : end].mean(-1)
            )
        cax = inset_axes(axes[1, 0], width="30%", height=1.0, loc=3)
        plt.colorbar(
            kl_colors_scalarMap,
            cax=cax,
            ticks=range(1, num_timesteps, 10),
            orientation="vertical",
        )

        plt.savefig(os.path.join(save_base, "latest_log.png"))
        plt.close("all")

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.mdl"))
