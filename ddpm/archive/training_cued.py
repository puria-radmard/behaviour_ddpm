import os
import torch
import numpy as np
from tqdm import tqdm

try:
    from sampling_ddpm.ddpm.model import *
    from sampling_ddpm.ddpm.simple_ddpm_tasks import MANIFOLD_INFORMATION_DICTIONARY
    from sampling_ddpm.ddpm.utils import symmetrize_and_square_axis
    from sampling_ddpm.ddpm.contextual_ddpm_tasks import generate_clifford_torus_directions, embed_2D_items_in_clifford
except ImportError:
    from ddpm.model import *
    from ddpm.simple_ddpm_tasks import MANIFOLD_INFORMATION_DICTIONARY
    from ddpm.utils import symmetrize_and_square_axis
    from ddpm.contextual_ddpm_tasks import generate_clifford_torus_directions, embed_2D_items_in_clifford

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer

from matplotlib import pyplot as plt


device = 'cuda'

num_items = 2

state_space_size = (2 + (2 * num_items)) * 10
time_embedding_size = 32
recurrence_hidden_layers = [state_space_size * 2, state_space_size * 2, state_space_size * 2]

stim_radius = 10.0

ultimate_sigma = 0.8 
starting_sigma = 0.1
num_timesteps = 100
noise_schedule_power = 4.0

sigma2x_schedule = (ultimate_sigma - starting_sigma) * 0.5 * (1.0 - torch.linspace(0, torch.pi, num_timesteps).cos())
sigma2x_schedule = (starting_sigma + sigma2x_schedule)**noise_schedule_power
sigma2x_schedule = sigma2x_schedule.to(device=device)

selected_timesteps = [0, 19, 39, 59, 79, 99]
num_noised_example_marginals = 5


save_base = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_cued_residual_21_12_24/run"
[training_print_path], save_base, _ = configure_logging_paths(save_base, log_suffixes=[f"train"], index_new=True)

logging_freq = 1000
magma = plt.get_cmap('magma')
cNorm  = colors.Normalize(vmin=1, vmax=num_timesteps)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])


input_size = 2 * num_items + num_items
input_model = InputModelBlock((input_size,), input_size)
residual_model = ResidualModel(state_space_size, recurrence_hidden_layers, input_size, time_embedding_size)
ddpm_model = DDPMReverseProcess(state_space_size, residual_model, input_model, sigma2x_schedule, time_embedding_size).to('cuda')


batch_size = 2048
num_trials = int(1e7)
lr = 1e-4

optim = torch.optim.Adam(ddpm_model.parameters(), lr = lr)
timer = LoopTimer(num_trials)

all_individual_residual_mses = np.zeros([num_trials, len(sigma2x_schedule)])


plt.plot(ddpm_model.sigma2xt_schedule.cpu().numpy(), label = 'sigma2xt_schedule', alpha = 0.4)
plt.plot(ddpm_model.a_t_schedule.cpu().numpy(), label = 'a_t_schedule', alpha = 0.4)
plt.plot(ddpm_model.root_b_t_schedule.cpu().numpy(), label = 'root_b_t_schedule', alpha = 0.4)
plt.plot(ddpm_model.noise_scaler_schedule.cpu().numpy(), label = 'noise_scaler_schedule', alpha = 0.4)
plt.plot(ddpm_model.base_samples_scaler_schedule.cpu().numpy(), label = 'base_samples_scaler_schedule', alpha = 0.4)
plt.plot(ddpm_model.residual_scaler_schedule.cpu().numpy(), label = 'residual_scaler_schedule', alpha = 0.4)
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


torus_directions = generate_clifford_torus_directions(ddpm_model, num_items)


for t in tqdm(range(num_trials)):

    timer.loop_start()

    training_sample_set = embed_2D_items_in_clifford(batch_size, torus_directions, num_items, stim_radius)  # [batch_size, 2]
    y_samples = training_sample_set['y_samples'].to(device=device, dtype = sigma2x_schedule.dtype)

    network_inputs = torch.cat([
        torch.nn.functional.one_hot(training_sample_set['network_inputs_idx'].to(device).to(torch.int64), num_classes=num_items), 
        # training_sample_set['network_inputs'].to(device),
        # training_sample_set['probe_clifford'].unsqueeze(0).repeat(batch_size, 1).to(device), 
        training_sample_set['report_clifford'].unsqueeze(0).repeat(batch_size, 1).to(device)
    ], -1)

    noised_examples = ddpm_model.noise(y_samples = y_samples)
    training_info = ddpm_model.residual(
        x_samples = noised_examples['x_t'],
        network_input = network_inputs,
        epsilon = noised_examples['epsilon']
    )
    scaled_mse = training_info['scaled_mse']

    optim.zero_grad()
    total_loss = scaled_mse.mean()
    total_loss.backward()
    optim.step()

    all_individual_residual_mses[t,:] = scaled_mse.detach().cpu().mean(0)

    if (t + 1850) % logging_freq == 0:
        
        novel_samples_dict = ddpm_model.generate_samples(
            network_input = network_inputs,
            # base_samples = training_sample_set['base_samples'] if isinstance(ddpm_model, DoublyConditionedDDPMReverseProcess) else None,
            samples_shape = (batch_size, ),# if isinstance(ddpm_model, DDPMReverseProcess) else None,
            turn_off_noise = False
        )
        novel_samples = novel_samples_dict['behaviour_samples'].detach().cpu().numpy()
        sample_trajectory = novel_samples_dict['sample_trajectory'].detach().cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        assert ddpm_model.sample_space_size == 2

        axes[0,0].scatter(*training_sample_set['y_samples'].T, c = training_sample_set['network_inputs_idx'])
        axes[0,0].add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))

        axes[0,1].add_patch(plt.Circle((0, 0), stim_radius, color='red', fill = False))
        axes[0,2].add_patch(plt.Circle((0, 0), stim_radius, color='red', fill = False))
        for m in range(num_items):
            axes[0,1].scatter(*(training_sample_set['base_samples'] @ torus_directions['probe_directions'].T)[:,[2 * m, 2 * m + 1]].T.cpu().numpy())
            axes[0,2].scatter(*(training_sample_set['base_samples'] @ torus_directions['report_directions'].T)[:,[2 * m, 2 * m + 1]].T.cpu().numpy())

        axes[0,0].set_title('x_0 samples')
        axes[0,0].legend(title = 'colours = different cues')
        axes[0,1].set_title('M (probe)')
        axes[0,1].legend(title = 'colours = different kernel subspaces')
        axes[0,2].set_title('M (report)')


        axes[1,0].scatter(novel_samples[:, 0], novel_samples[:, 1], c = training_sample_set['network_inputs_idx'], alpha=0.5, s=1)
        axes[1,0].add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))
        axes[1,0].set_title('novel samples, colored by Z')

        projected_trajs = sample_trajectory[:5] @ ddpm_model.linking_matrix.T.detach().cpu().numpy()
        for ex in range(5):
            axes[1,0].plot(*projected_trajs[ex].T, color = 'green', alpha = 0.1)
        
        near_end_step_samples = sample_trajectory[:,-2] @ ddpm_model.linking_matrix.T.detach().cpu().numpy()
        axes[1,1].scatter(near_end_step_samples[:, 0], near_end_step_samples[:, 1], c = training_sample_set['network_inputs_idx'], alpha=0.5, s=1)
        axes[1,1].add_patch(plt.Circle((0, 0), 1.0, color='red', fill = False))
        axes[1,1].set_title('novel samples a few steps before end, colored by Z')
            
        for h, trace in enumerate(all_individual_residual_mses[:t+1].T):
            color = kl_colors_scalarMap.to_rgba(h + 1)
            axes[1,2].plot(trace, color = color)

        symmetrize_and_square_axis(axes[0,0], 2.0)
        symmetrize_and_square_axis(axes[0,1], stim_radius + 2.0)
        symmetrize_and_square_axis(axes[0,2], stim_radius + 2.0)
        symmetrize_and_square_axis(axes[1,0], 2.0)
        symmetrize_and_square_axis(axes[1,1], 2.0)

        cax = inset_axes(axes[1,2],
                    width="30%", # width = 30% of parent_bbox
                    height=1., # height : 1 inch
                    loc=3)
        plt.colorbar(kl_colors_scalarMap, cax = cax, ticks=range(1, num_timesteps, 10), orientation='vertical')

        plt.savefig(os.path.join(save_base, f"log_T{num_timesteps}.png"))
        plt.close('all')

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state_T{num_timesteps}.mdl"))

