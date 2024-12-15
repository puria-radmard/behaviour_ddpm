raise Exception('Abandoning DNG for now - see unrolling training')

import sys
import os
from tqdm import tqdm
import numpy as np
import random

from sklearn.decomposition import PCA

from sampling_ddpm.utils import *
from sampling_ddpm.model import *
from sampling_ddpm.mog_2d import generate_random_gaussian_mixture

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer

from scipy.stats import iqr
from statistics import median

import argparse
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


parser = argparse.ArgumentParser()

parser.add_argument('--save_base', type = str, required = False, default = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_dng_test_11_11_24")
parser.add_argument('--run_name', type = str, required = True)
parser.add_argument('--hidden_size', type = int, required = True)
parser.add_argument('--num_trials', type = int, required = False, default = 100_000)
parser.add_argument('--resume_previous_flag', required = False, action = 'store_true')

args = parser.parse_args()

num_trials = args.num_trials
hidden_size = args.hidden_size
resume_previous_flag = args.resume_previous_flag

device = 'cuda'

num_modes = 4
num_hidden_steps = 1   # T
sigma2x_schedule = 0.1 * torch.ones(num_hidden_steps) # 0.1 + 0.9 / (1 + (-(torch.arange(num_hidden_steps) - (num_hidden_steps/2)) / 10.0).exp())
sigma2x_schedule = sigma2x_schedule.to(device=device)
noise_model_hidden_layers = [512]
recurrence_hidden_layers = [hidden_size * 2, hidden_size * 2]


magma = plt.get_cmap('magma')
cNorm  = colors.Normalize(vmin=1, vmax=num_hidden_steps)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])



def get_upper_ol_limit(arr):
    arr = np.nan_to_num(arr.flatten())
    return median(arr) + 1.5 * iqr(arr)

def symmetrize_and_square_axis(axes, min_size = None):
    x_max = np.max(np.abs(axes.get_ylim()))
    y_max = np.max(np.abs(axes.get_ylim()))
    xy_max = max(x_max, y_max)
    if min_size is not None:
        xy_max = max(xy_max, min_size)
    axes.set_ylim(ymin=-xy_max, ymax=xy_max)
    axes.set_xlim(xmin=-xy_max, xmax=xy_max)



if resume_previous_flag:
    raise NotImplementedError
    save_base = os.path.join(args.save_base, args.run_name)
    noise_model = TimeSensitiveNoiseProcessCovarGenerator(noise_model_hidden_layers, hidden_size - 1)
    old_ddpm_model = DDPMReverseProcess(hidden_size - 1, sigma2x_schedule, noise_model)
    old_ddpm_model.load_state_dict(torch.load(os.path.join(save_base, f"state_{hidden_size - 1}.mdl")))
    random_neuron = random.randint(0, hidden_size - 2)
    print(f'Splitting neuron {random_neuron}')
    ddpm_model = old_ddpm_model.do_dng(random_neuron).to(device=device)
    training_print_path = os.path.join(save_base, f"epoch_log_train_{hidden_size}.csv")

else:
    save_base = os.path.join(args.save_base, args.run_name)
    [training_print_path], save_base, _ = configure_logging_paths(save_base, log_suffixes=[f"train_{hidden_size}"], index_new=True)
    noise_model = TimeConstantNoiseProcessCovarGenerator(hidden_size)
    recurrence_model = RecurrenceModel(hidden_size, recurrence_hidden_layers)
    ddpm_model = DDPMReverseProcess(hidden_size, sigma2x_schedule, noise_model, recurrence_model, tie_behvaiour_projection_matrix = False).to(device=device)



with open(training_print_path, 'a') as f:
    header_row = [
        "Progress",
        "avg_recon_loss",
    ] + [
        f"avg_kl_loss_{i}" for i in range(1, num_hidden_steps)
    ]
    print(*header_row, "elapsed", "remaining", sep = '\t', file=f)


all_total_losses = np.zeros(num_trials)
all_individual_kl_losses = np.zeros([num_trials, len(sigma2x_schedule) - 1])
all_recon_losses = np.zeros(num_trials)

optim = torch.optim.Adam(ddpm_model.parameters(), lr = 1e-4)

timer = LoopTimer(num_trials)

for t in tqdm(range(num_trials)):

    timer.loop_start()

    y_samples = generate_random_gaussian_mixture(2048, num_modes).to(device=device)  # [2048, 2]
    x_samples = run_forward_process(y_samples, sigma2x_schedule, ddpm_model.linking_matrix)   # [2048, T, hidden_size]

    one_step_reverse_process_dict = ddpm_model.reverse_generative_process(x_samples)
    q_reversed_mu_ts, q_reversed_Sigma_ts = generate_reversed_noising_posteriors(y_samples, x_samples, sigma2x_schedule, ddpm_model.linking_matrix)

    loss = cross_entropy_terms_on_reverse_process(
        y_samples = y_samples,
        q_reversed_mu_ts = q_reversed_mu_ts,
        q_reversed_Sigma_ts = q_reversed_Sigma_ts,
        **one_step_reverse_process_dict,
        forward_kl = False
    )

    optim.zero_grad()

    total_loss = 0.0
    for k, v in loss.items():
        total_loss = total_loss + v.sum() / v.shape[0]

    total_loss.backward()
    optim.step()

    all_individual_kl_losses[t,:] = loss['all_kl_terms'].detach().cpu().mean(0)
    all_recon_losses[t] = loss['reconstruction_cross_entropy'].detach().cpu().mean()
    all_total_losses[t] = total_loss.item()

    if (t + 1) % 500 == 0:
        fig, axes = plt.subplots(4, 2, figsize=(12, 24))

        y_samples = y_samples.cpu().numpy()
        axes[0,0].scatter(y_samples[:, 0], y_samples[:, 1], alpha=0.5, s=1)
        axes[0,0].set_title('y samples (ground truth)')

        pca_final_x = PCA(2).fit_transform(x_samples[:,-1].detach().cpu().numpy())
        axes[0,1].scatter(pca_final_x[:,0], pca_final_x[:,1], alpha=0.5, s=1)
        axes[0,1].set_title('Full noising of GT (PCA)')
                
        cl = 5.0
        for ax in axes[0]:
            ax.set_xlim(-cl, cl)
            ax.set_ylim(-cl, cl)

        y_recreated_means = one_step_reverse_process_dict['y_mu_p'].detach().cpu().numpy()
        axes[1,0].scatter(y_recreated_means[:, 0], y_recreated_means[:, 1], alpha=0.5, s=1)
        axes[1,0].set_title('One step y recreation means')

        with torch.no_grad():
            novel_samples = ddpm_model.generate_new_samples(
                num_samples = 2048,
                num_hidden_steps = num_hidden_steps,
            ).detach().cpu().numpy()
        axes[1,1].scatter(novel_samples[:, 0], novel_samples[:, 1], alpha=0.5, s=1)
        axes[1,1].set_title('novel samples')

        symmetrize_and_square_axis(axes[1,0], 5.0)
        symmetrize_and_square_axis(axes[1,1], 5.0)

        for ax_row in axes[[2,3]]:
            ax_row[0].plot(all_total_losses[:t+1])
            ax_row[0].set_title('Total loss')

            alpha_offset = 1
            for h, trace in enumerate(all_individual_kl_losses[:t+1].T):
                color = kl_colors_scalarMap.to_rgba(h + 1)
                ax_row[1].plot(trace, color = color)
            
            lossax2 = ax_row[1].twinx()
            lossax2.plot(all_recon_losses[:t+1], color = 'red', alpha = 0.3)
            lossax2.set_ylabel('', color='red')  # we already handled the x-label with ax1
            lossax2.tick_params(axis='y', labelcolor='red')        
            ax_row[1].set_title('Magma = kl losses\nRed = reconstruction loss')

        axes[3,0].set_ylim(np.nanmin(all_total_losses[:t+1]), get_upper_ol_limit(all_total_losses[:t+1]))
        if num_hidden_steps > 1:
            axes[3,1].set_ylim(np.nanmin(all_individual_kl_losses[:t+1]), get_upper_ol_limit(all_individual_kl_losses[:t+1]))
        lossax2.set_ylim(np.nanmin(all_recon_losses[:t+1]), get_upper_ol_limit(all_recon_losses[:t+1]))

        cax = inset_axes(axes[2,1],
                    width="30%", # width = 30% of parent_bbox
                    height=1., # height : 1 inch
                    loc=3)
        plt.colorbar(kl_colors_scalarMap, cax = cax, ticks=range(1, num_hidden_steps, 10), orientation='vertical')

        plt.savefig(os.path.join(save_base, f"log_{hidden_size}.png"))
        plt.close('all')
        
        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state_{hidden_size}.mdl"))


    elapsed_string, remaining_string = timer.loop_end()
    with open(training_print_path, 'a') as f:
        new_print_row = [
            f"{t + 1}/{num_trials}",                                                                 # Progress
            round(all_recon_losses[t], 6),
        ] + [
            all_individual_kl_losses[t,i] for i in range(num_hidden_steps-1)
        ]
        new_print_row.extend([elapsed_string, remaining_string])
        print(*new_print_row, sep = '\t', file=f)

