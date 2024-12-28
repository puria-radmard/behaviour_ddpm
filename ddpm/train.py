import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from purias_utils.util.arguments_yaml import ConfigNamepace

from ddpm.model import DDPMReverseProcessBase
from ddpm.utils import symmetrize_and_square_axis
from ddpm import tasks, model

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer

from matplotlib import pyplot as plt


args = ConfigNamepace.from_yaml_path(sys.argv[1])


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Unpack model and diffusion args
state_space_size = args.state_space_size
recurrence_hidden_layers = args.recurrence_hidden_layers
time_embedding_size = args.time_embedding_size
ultimate_sigma2 = args.ultimate_sigma2
starting_sigma2 = args.starting_sigma2
num_timesteps = args.num_timesteps
noise_schedule_power = args.noise_schedule_power


# Generate noising schedule
# sigma2x_schedule = (ultimate_sigma2 - starting_sigma2) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
# sigma2x_schedule = (starting_sigma2 + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_timesteps)
sigma2x_schedule = sigma2x_schedule.to(device=device)



# Set up task
task_name = args.task_name
task_config = args.task_config
task: tasks.WMDiffusionTask = getattr(tasks, task_name)(**task_config.dict)


# Set up model
model_name = args.model_name
model_config = args.model_config
ddpm_model: DDPMReverseProcessBase = getattr(model, model_name)(
    **model_config.dict, 
    sigma2x_schedule = sigma2x_schedule, sensory_shape = task.sensory_gen.sensory_shape, sample_shape = task.sample_gen.sample_shape,
    device = device
).to(device)


# Set up training
batch_size = args.batch_size
num_trials = args.num_trials
lr = args.lr
optim = torch.optim.Adam(ddpm_model.parameters(), lr = lr)


# Set up logging
logging_freq = args.logging_freq
save_base = args.save_base
magma = plt.get_cmap('magma')
cNorm  = colors.Normalize(vmin=1, vmax=num_timesteps)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])
timer = LoopTimer(num_trials)
[training_print_path], save_base, _ = configure_logging_paths(save_base, log_suffixes=[f"train"], index_new=True)
all_individual_residual_mses = np.zeros([num_trials, len(sigma2x_schedule)])
args.write_to_yaml(os.path.join(save_base, 'args.yaml'))
task.save_metadata(os.path.join(save_base, 'task_metadata'))


# For transparency
sch_fig, sch_axes = plt.subplots(2, 1, figsize = (5, 8))
sch_axes[0].plot(ddpm_model.sigma2xt_schedule.cpu().numpy(), label = 'sigma2xt_schedule', alpha = 0.4)
sch_axes[0].plot(ddpm_model.a_t_schedule.cpu().numpy(), label = 'a_t_schedule', alpha = 0.4)
sch_axes[0].plot(ddpm_model.root_b_t_schedule.cpu().numpy(), label = 'root_b_t_schedule', alpha = 0.4)
sch_axes[0].plot(ddpm_model.noise_scaler_schedule.cpu().numpy(), label = 'noise_scaler_schedule', alpha = 0.4)
sch_axes[0].plot(ddpm_model.base_samples_scaler_schedule.cpu().numpy(), label = 'base_samples_scaler_schedule', alpha = 0.4)
sch_axes[0].plot(ddpm_model.residual_scaler_schedule.cpu().numpy(), label = 'residual_scaler_schedule', alpha = 0.4)
sch_axes[0].legend()

sch_axes[1].set_title('Time embeddings')
sch_axes[1].imshow(ddpm_model.time_embeddings.time_embs.detach().cpu().numpy().T)

plt.savefig(os.path.join(save_base, "sigma_schedule_unrolling.png"))



# This will get filled in and continuously updated by task.sample_gen.generate_sample_diagnostics
recent_sample_diagnostics = deque(maxlen=100)


for t in tqdm(range(num_trials)):

    timer.loop_start()

    trial_information = task.generate_trial_information(num_samples=batch_size)

    forward_process = ddpm_model.noise(x_0 = trial_information.sample_information.sample_set.to(device).float())
    epsilon_hat = ddpm_model.residual(
        x_samples = forward_process['x_t'],
        network_input = trial_information.network_inputs,
    )
    residual_mse = task.sample_gen.mse(epsilon_hat, forward_process['epsilon'])

    optim.zero_grad()
    total_loss = residual_mse.mean()
    total_loss.backward()
    optim.step()

    all_individual_residual_mses[t,:] = residual_mse.detach().cpu().mean(0)

    if (t - 10) % logging_freq == 0:

        novel_samples_dict = ddpm_model.generate_samples(
            network_input = trial_information.network_inputs,
            num_samples = batch_size,
            turn_off_noise = False
        )

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        axes[0,0].set_title('Real sample(s)')
        axes[0,1].set_title('Generated sample(s)')
        axes[0,2].set_title('Early predictions of $x_0$')
        axes[1,2].set_title('Samples from base distribution')
        task.sample_gen.display_samples(trial_information.sample_information, axes[0,0])
        task.sample_gen.display_samples(novel_samples_dict['samples'], axes[0,1])
        task.sample_gen.display_samples(forward_process['x_t'][:,-1], axes[1,2])
        task.sample_gen.display_early_x0_pred_timeseries(novel_samples_dict['early_x0_preds'], axes[0,2], kl_colors_scalarMap)

        task.task_variable_gen.display_task_variables(trial_information.task_variable_information, axes[1,0], axes[1,1])
        
        symmetrize_and_square_axis(axes[1,0])
        symmetrize_and_square_axis(axes[1,1])
        symmetrize_and_square_axis(axes[1,2])
        symmetrize_and_square_axis(axes[0,0])
        symmetrize_and_square_axis(axes[0,1])
        symmetrize_and_square_axis(axes[0,2])
        
        # recent_sample_diagnostics = task.sample_gen.generate_sample_diagnostics(
        #     sample_set = novel_samples_dict, variables_dict = trial_information, 
        #     recent_sample_diagnostics = recent_sample_diagnostics, axes = axes[2,0]
        # )

        for h, trace in enumerate(all_individual_residual_mses[:t+1].T):
            color = kl_colors_scalarMap.to_rgba(h + 1)
            axes[2,1].plot(trace, color = color)
        cax = inset_axes(axes[2,0], width="30%", height=1.,loc=3)
        plt.colorbar(kl_colors_scalarMap, cax = cax, ticks=range(1, num_timesteps, 10), orientation='vertical')

        plt.savefig(os.path.join(save_base, "latest_log.png"))
        if (t + 1850) % (logging_freq*100) == 0:
            plt.savefig(os.path.join(save_base, f"log_{t}.png"))
        plt.close('all')

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state_T{num_timesteps}.mdl"))
