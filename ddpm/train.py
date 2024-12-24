import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from purias_utils.util.arguments_yaml import ConfigNamepace

try:
    from sampling_ddpm.ddpm.model import *
    from sampling_ddpm.ddpm.utils import symmetrize_and_square_axis
    from sampling_ddpm.ddpm.contextual_ddpm_tasks import generate_clifford_torus_directions, embed_2D_items_in_clifford
    from sampling_ddpm.ddpm import tasks
except ImportError:
    from ddpm.model import *
    from ddpm.utils import symmetrize_and_square_axis
    from ddpm.contextual_ddpm_tasks import generate_clifford_torus_directions, embed_2D_items_in_clifford
    from ddpm import tasks

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
ultimate_sigma = args.ultimate_sigma
starting_sigma = args.starting_sigma
num_timesteps = args.num_timesteps
noise_schedule_power = args.noise_schedule_power


# Generate noising schedule
sigma2x_schedule = (ultimate_sigma - starting_sigma) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
sigma2x_schedule = (starting_sigma + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = sigma2x_schedule.to(device=device)



# Set up task
task_name = args.task_name
task_config = args.task_config
task: tasks.WMDiffusionTask = getattr(tasks, task_name)(**task_config)



# Set up model
input_shape = task.input_gen.input_shape
if args.input_type == 'tabular':
    input_model = InputModelBlock(input_shape, input_shape[0])
residual_model = ResidualModel(state_space_size, recurrence_hidden_layers, input_shape[0], time_embedding_size)
ddpm_model = DDPMReverseProcess(state_space_size, residual_model, input_model, sigma2x_schedule, time_embedding_size).to('cuda')


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


# This will get filled in and continuously updated by task.sample_gen.generate_sample_diagnostics
recent_sample_diagnostics = deque(maxlen=100)


for t in tqdm(range(num_trials)):

    timer.loop_start()

    trial_information = task.generate_trial_information(num_samples=batch_size)
    
    forward_process = ddpm_model.noise(y_samples = trial_information.sample_set)
    training_info = ddpm_model.residual(
        x_samples = forward_process['x_t'],
        network_input = trial_information.network_inputs,
        epsilon = forward_process['epsilon']
    )
    residual_mse = training_info['scaled_mse']

    optim.zero_grad()
    total_loss = residual_mse.mean()
    total_loss.backward()
    optim.step()

    all_individual_residual_mses[t,:] = residual_mse.detach().cpu().mean(0)

    if (t + 1850) % logging_freq == 0:

        with torch.no_grad():
            novel_samples_dict = ddpm_model.generate_samples(
                network_input = trial_information.network_inputs,
                samples_shape = (batch_size, ),
                turn_off_noise = False
            )

        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        
        axes[0,0].set_title('Real sample(s)')
        axes[0,1].set_title('Generated sample(s)')
        task.sample_gen.display_samples(trial_information.sample_set, axes[0,0])
        task.sample_gen.display_samples(novel_samples_dict['behaviour_samples'], axes[0,0])
        task.task_variable_gen.display_task_variables(trial_information.task_variable_information, axes[1,0], axes[1,1])
        
        recent_sample_diagnostics = task.sample_gen.generate_sample_diagnostics(
            sample_set = novel_samples_dict, variables_dict = trial_information, 
            recent_sample_diagnostics = recent_sample_diagnostics, axes = axes[2,0]
        )

        for h, trace in enumerate(all_individual_residual_mses[:t+1].T):
            color = kl_colors_scalarMap.to_rgba(h + 1)
            axes[2,1].plot(trace, color = color)
        cax = inset_axes(axes[1,1], width="30%", height=1.,loc=3)

        plt.savefig(os.path.join(save_base, "latest_log.png"))
        if ((t + 1850) / 100) % logging_freq:
            plt.savefig(os.path.join(save_base, f"log_{t}.png"))
        plt.close('all')

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state_T{num_timesteps}.mdl"))
