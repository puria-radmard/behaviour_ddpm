import torch
from torch import nn

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sampling_ddpm.ddpm.utils import symmetrize_and_square_axis
from sampling_ddpm.ddpm.model import VectoralResidualModel, DDPMReverseProcess, InputModelBlock
from sampling_ddpm.ddpm.simple_ddpm_tasks import MANIFOLD_INFORMATION_DICTIONARY

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



# task_name = 'simple_singleitem'
task_name = 'simple_multiitem'
data_generation_function, manifold_x, manifold_y, stimulus_shape = MANIFOLD_INFORMATION_DICTIONARY[task_name]

state_space_size = 16
time_embedding_size = 32
stimulus_representation_size = stimulus_shape[0]
recurrence_hidden_layers = [128, 128, 128]

ultimate_sigma = 0.8 
starting_sigma = 0.1
num_timesteps = 100
noise_schedule_power = 4.0





save_base = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_residual_15_12_24/run"
[training_print_path], save_base, _ = configure_logging_paths(save_base, log_suffixes=[f"train"], index_new=True)

logging_freq = 1000
magma = plt.get_cmap('magma')
cNorm  = colors.Normalize(vmin=1, vmax=num_timesteps)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])

device = 'cuda'




sigma2x_schedule = (ultimate_sigma - starting_sigma) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
sigma2x_schedule = (starting_sigma + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = sigma2x_schedule.to(device=device)


input_model = InputModelBlock(stimulus_shape, stimulus_representation_size)
residual_model = VectoralResidualModel(state_space_size, recurrence_hidden_layers, stimulus_representation_size, time_embedding_size)
ddpm_model = DDPMReverseProcess(16, residual_model, input_model, sigma2x_schedule, time_embedding_size).to('cuda')

# ddpm_model.load_state_dict(
#     torch.load('/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_residual_15_12_24/run_8/state_T100.mdl')
# )


batch_size = 2048
num_trials = int(1e7)
lr = 1e-4

optim = torch.optim.Adam(ddpm_model.parameters(), lr = lr)
timer = LoopTimer(num_trials)

#plt.plot(sigma2x_schedule.cpu(), color = 'blue', alpha = 0.1, zorder= 0)

plt.plot(ddpm_model.sigma2xt_schedule.cpu().numpy(), label = 'sigma2xt_schedule', alpha = 0.4)
plt.plot(ddpm_model.a_t_schedule.cpu().numpy(), label = 'a_t_schedule', alpha = 0.4)
plt.plot(ddpm_model.root_b_t_schedule.cpu().numpy(), label = 'root_b_t_schedule', alpha = 0.4)
plt.plot(ddpm_model.noise_scaler_schedule.cpu().numpy(), label = 'noise_scaler_schedule', alpha = 0.4)
plt.plot(ddpm_model.base_samples_scaler_schedule.cpu().numpy(), label = 'base_samples_scaler_schedule', alpha = 0.4)
plt.plot(ddpm_model.residual_scaler_schedule.cpu().numpy(), label = 'residual_scaler_schedule', alpha = 0.4)
# plt.plot(ddpm_model.mse_scaler_schedule.cpu().numpy(), label = 'mse_scaler_schedule', alpha = 0.4)
plt.legend()
plt.savefig(os.path.join(save_base, "sigma_schedule_unrolling.png"))



with open(training_print_path, 'w') as f:
    header_row = [
        "Progress",
        "lr",
        "avg_recon_loss",
    ] + [
        f"avg_kl_loss_{i}" for i in range(1, num_timesteps)
    ]
    print(*header_row, "elapsed", "remaining", sep = '\t', file=f)


all_individual_residual_mses = np.zeros([num_trials, len(sigma2x_schedule)])


for t in tqdm(range(num_trials)):

    timer.loop_start()

    if t >= 0:
        y_samples, network_inputs, network_input_classes = data_generation_function(batch_size, True)  # [batch_size, 2]
        y_samples = y_samples.to(device=device, dtype = sigma2x_schedule.dtype)

    noised_examples = ddpm_model.noise(y_samples)
    training_info = ddpm_model.residual(noised_examples['x_t'], network_input = network_inputs, epsilon = noised_examples['epsilon'])
    scaled_mse = training_info['scaled_mse']

    optim.zero_grad()
    total_loss = scaled_mse.mean()
    total_loss.backward()
    optim.step()

    all_individual_residual_mses[t,:] = scaled_mse.detach().cpu().mean(0)

    if (t + 1850) % logging_freq == 0:

        with torch.no_grad():
            novel_samples_dict= ddpm_model.generate_samples(
                network_input = network_inputs, samples_shape = (batch_size,), turn_off_noise=False
            )
            novel_samples = novel_samples_dict['behaviour_samples'].detach().cpu().numpy()
            sample_trajectory = novel_samples_dict['sample_trajectory'].detach().cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        assert ddpm_model.sample_space_size == 2

        y_samples_np = y_samples.cpu().numpy()
        axes[0,0].scatter(y_samples_np[:, 0], y_samples_np[:, 1], alpha=0.5, s=1, c=network_input_classes)
        axes[0,0].plot(manifold_x, manifold_y, alpha = 0.1, color = 'red')
        axes[0,0].set_title('y samples (ground truth)')

        axes[0,1].scatter(novel_samples[:, 0], novel_samples[:, 1], alpha=0.5, s=1)
        axes[0,1].plot(manifold_x, manifold_y, alpha = 0.1, color = 'red')
        axes[0,1].set_title('novel samples')

        projected_trajs = sample_trajectory[:5] @ ddpm_model.linking_matrix.T.detach().cpu().numpy()
        for ex in range(5):
            axes[0,1].plot(*projected_trajs[ex].T, color = 'green', alpha = 0.1)
            

        for h, trace in enumerate(all_individual_residual_mses[:t+1].T):
            color = kl_colors_scalarMap.to_rgba(h + 1)
            axes[1,0].plot(trace, color = color)

        symmetrize_and_square_axis(axes[0,0], 2.0)
        symmetrize_and_square_axis(axes[0,1], 2.0)

        cax = inset_axes(axes[1,0],
                    width="30%", # width = 30% of parent_bbox
                    height=1., # height : 1 inch
                    loc=3)
        plt.colorbar(kl_colors_scalarMap, cax = cax, ticks=range(1, num_timesteps, 10), orientation='vertical')

        plt.savefig(os.path.join(save_base, f"log_T{num_timesteps}.png"))
        plt.close('all')
        
        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state_T{num_timesteps}.mdl"))
