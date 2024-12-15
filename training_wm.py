import os
from tqdm import tqdm
import numpy as np
import json

from sklearn.decomposition import PCA

from sampling_ddpm.utils import *
from sampling_ddpm.model import *
from sampling_ddpm.simple_ddpm_structured_tasks import WM_TASK_INFORMATION_DICTIONARY

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

parser.add_argument('--save_base', type = str, required = False, default = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_unrolling_circle_28_11_24")
parser.add_argument('--run_name', type = str, required = True)
parser.add_argument('--num_trials', type = int, required = False, default = 2_000_000)
parser.add_argument('--resume_path', type = str, required = False)

parser.add_argument('--manifold_name', type = str, required = True, choices=WM_TASK_INFORMATION_DICTIONARY.keys())

parser.add_argument('--hidden_size', type = int, required = True)
parser.add_argument('--recurrence_hidden_layers', type = int, nargs = '+', required = False, default = [64, 64, 64, 64])

parser.add_argument('--lr', type = float, required = True)
parser.add_argument('--lr_reduce_factor', type = float, required = False, default = 1.5)
parser.add_argument('--lr_reduce_trials', type = int, required = False, default = 3_000_000)

parser.add_argument('--baseline_sigma2', type = float, required = False, default = 0.1)
parser.add_argument('--ultimate_sigma2', type = float, required = False, default = 1.0)
parser.add_argument('--sigma2x_orthogonal_multiplier', type = float, required = False, default = 5.0)

# Ult timesteps taken out! Not really needed...
parser.add_argument('--num_timesteps', type = int, nargs='+', required = True, help = 'going backwards, i.e. num recall steps, then num wm steps, then num loading steps')
parser.add_argument('--sigma2s', type = float, nargs='+', required = True)
parser.add_argument('--noise_schedule_power', nargs='+', type = float, required = True)

args = parser.parse_args()


data_generation_function, num_epochs, input_size = WM_TASK_INFORMATION_DICTIONARY[args.manifold_name]
assert len(args.num_timesteps) == len(args.num_timesteps) == len(args.sigma2s) - 1



logging_freq = 2000

batch_size = 8192



device = 'cuda'

all_epoch_sigma2x_schedules = []    # still going backwards!!

for sigma2_start, sigma2_end, curr_epoch_steps, noise_schedule_power, in zip(
    args.sigma2s[:-1], args.sigma2s[1:], args.num_timesteps, args.noise_schedule_power
):
    epoch_sigma2x_schedule_evo = (sigma2_end - sigma2_start) * 0.5 * (1.0 - torch.linspace(0, torch.pi, curr_epoch_steps).cos())
    epoch_sigma2x_schedule = sigma2_start + epoch_sigma2x_schedule_evo**noise_schedule_power
    epoch_sigma2x_schedule = epoch_sigma2x_schedule.to(device=device)
    all_epoch_sigma2x_schedules.append(epoch_sigma2x_schedule)



magma = plt.get_cmap('magma')
cNorm  = colors.Normalize(vmin=1, vmax=sum(args.num_timesteps))
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])



def get_upper_ol_limit(arr):
    arr = np.nan_to_num(arr.flatten())
    return median(arr) + 1.5 * iqr(arr)

def symmetrize_and_square_axis(axes, min_size = None):
    x_max = np.max(np.abs(axes.get_xlim()))
    y_max = np.max(np.abs(axes.get_ylim()))
    xy_max = max(x_max, y_max)
    if min_size is not None:
        xy_max = max(xy_max, min_size)
    axes.set_ylim(ymin=-xy_max, ymax=xy_max)
    axes.set_xlim(xmin=-xy_max, xmax=xy_max)


recurrence_model = RecurrenceModelWithInput(hidden_size, recurrence_hidden_layers, input_size)
ddpm_model = DDPMReverseProcess(
    hidden_size,
    sigma2x_schedule,
    ultimate_time_steps,
    recurrence_model,
).to(device=device)

if args.resume_path:
    ddpm_model.load_state_dict(torch.load(args.resume_path))


save_base = os.path.join(args.save_base, args.run_name)

[training_print_path], save_base, _ = configure_logging_paths(save_base, log_suffixes=[f"train"], index_new=True)

# sum(p.numel() for p in recurrence_model.parameters() if p.requires_grad)

with open(os.path.join(save_base, f"args.json"), 'w') as jf:
    json.dump(vars(args), jf)

plt.plot(sigma2x_schedule.cpu(), color = 'blue')
plt.plot(full_sigma2x_schedule.cpu(), color = 'red', alpha = 0.1, zorder= 0)
plt.savefig(os.path.join(save_base, "sigma_schedule_unrolling.png"))


with open(training_print_path, 'w') as f:
    header_row = [
        "Progress",
        "lr",
        "avg_recon_loss",
    ] + [
        f"avg_kl_loss_{i}" for i in range(1, num_hidden_steps)
    ]
    print(*header_row, "elapsed", "remaining", sep = '\t', file=f)


all_total_losses = np.zeros(num_trials)
all_individual_kl_losses = np.zeros([num_trials, len(sigma2x_schedule) - 1])
all_recon_losses = np.zeros(num_trials)

# up_grad = UPGrad()
optim = torch.optim.Adam(ddpm_model.parameters(), lr = args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_reduce_trials, gamma=1. / args.lr_reduce_factor)
timer = LoopTimer(num_trials)

for t in tqdm(range(num_trials)):

    timer.loop_start()

    y_samples, network_inputs, network_input_classes = data_generation_function(batch_size)  # [batch_size, 2]
    y_samples = y_samples.to(device=device, dtype = sigma2x_schedule.dtype)
    x_samples = run_forward_process(y_samples, sigma2x_schedule, sigma2x_orthogonal_multiplier, ddpm_model.linking_matrix, ddpm_model.rotation_matrix)   # [batch_size, T, hidden_size]

    if network_inputs is not None:
        network_inputs = network_inputs.to(device)

    one_step_reverse_process_dict = ddpm_model.one_step_reverse_generative_process(x_samples, input_vector = network_inputs)
    q_reversed_z_mu_ts, q_reversed_z_Sigma_ts = generate_reversed_noising_posteriors(y_samples, x_samples, sigma2x_schedule, sigma2x_orthogonal_multiplier, ddpm_model.linking_matrix, ddpm_model.rotation_matrix)

    loss = cross_entropy_terms_on_reverse_process(
        y_samples = y_samples,
        q_reversed_z_mu_ts = q_reversed_z_mu_ts,
        q_reversed_z_Sigma_ts = q_reversed_z_Sigma_ts,
        **one_step_reverse_process_dict,
        projection_basis = ddpm_model.rotation_matrix
    )

    optim.zero_grad()

    total_loss = 0.0
    for k, v in loss.items():
        total_loss = total_loss + v.sum() / v.shape[0]
    total_loss.backward()
    
    # losses_as_tensors = [loss['reconstruction_cross_entropy'].mean(0)]
    # losses_as_tensors.extend(loss['all_kl_terms'].mean(0))
    # backward(
    #     tensors=losses_as_tensors,
    #     inputs=ddpm_model.parameters(),
    #     A=up_grad,
    # )

    optim.step()
    scheduler.step()

    all_individual_kl_losses[t,:] = loss['all_kl_terms'].detach().cpu().mean(0)
    all_recon_losses[t] = loss['reconstruction_cross_entropy'].detach().cpu().mean()
    all_total_losses[t] = total_loss.item()

    if (t) % logging_freq == 0:
        fig, axes = plt.subplots(4, 2, figsize=(12, 24)) if hidden_size == 2 else plt.subplots(4, 3, figsize=(18, 24))

        assert ddpm_model.sample_space_size == 2

        ## PLOT GROUND TRUTH
        y_samples = y_samples.cpu().numpy()
        axes[0,0].scatter(y_samples[:, 0], y_samples[:, 1], alpha=0.5, s=1, c=network_input_classes)
        axes[0,0].plot(manifold_x, manifold_y, alpha = 0.1, color = 'red')
        axes[0,0].set_title('y samples (ground truth)')

        ## PLOT FINAL NOISED VERSION IN PLANE OF BEHAVIOUR
        axes[0,1].set_title('Full noising of GT (projected to behaviour space)' if hidden_size > 2 else 'Full noising of GT')
        axes[0,1].plot(manifold_x, manifold_y, alpha = 0.1, color = 'red')
        final_x = (x_samples[:,-1] @ ddpm_model.behvaiour_projection_matrix).detach().cpu().numpy()
        axes[0,1].scatter(final_x[:,0], final_x[:,1], alpha=0.5, s=1, c=network_input_classes)

        ## PLOT ONE STEP RECREATIONS OF BEHAVIOUR
        y_recreated_means = one_step_reverse_process_dict['y_mu_p'].detach().cpu().numpy()
        axes[1,0].scatter(y_recreated_means[:, 0], y_recreated_means[:, 1], alpha=0.5, s=1, c=network_input_classes)
        axes[1,0].plot(manifold_x, manifold_y, alpha = 0.1, color = 'red')
        axes[1,0].set_title('One step y recreation means')

        ## PLOT FULL TRAJECTORIES IN PLANE OF BEHAVIOUR
        with torch.no_grad():
            novel_samples, hidden_sample_history, novel_x0 = ddpm_model.generate_new_samples(
                num_samples = batch_size,
                base_samples = x_samples[:,-1],
                input_vector = network_inputs
            )
            novel_samples = novel_samples.detach().cpu().numpy()
        selected_traces_in_behaviour_plane = hidden_sample_history[:5] @ ddpm_model.behvaiour_projection_matrix.cpu().numpy()
        for st_i, st in enumerate(selected_traces_in_behaviour_plane):
            axes[1, 1].plot(*st.T, color = 'green', alpha = 0.4, zorder = 100 + st_i)
        axes[1,1].scatter(novel_samples[:, 0], novel_samples[:, 1], alpha=0.5, s=1, c=network_input_classes)
        axes[1,1].plot(manifold_x, manifold_y, alpha = 0.1, color = 'red')
        axes[1,1].set_title('novel samples')

        ## PLOT FULL TRAJECTORIES ORTHOGONAL TO BEHAVIOUR
        if ddpm_model.hidden_size > 3:
            selected_traces_out_of_behaviour_plane = hidden_sample_history[:5] @ ddpm_model.linking_matrix_kernel.T.cpu().numpy()
            novel_samples_out_of_behaviour_plane = (novel_x0 @ ddpm_model.linking_matrix_kernel.T).detach().cpu().numpy()
            pca_transformer = PCA(n_components=2).fit(novel_samples_out_of_behaviour_plane)
            pca_novel_samples_out_of_behaviour_plane = pca_transformer.transform(novel_samples_out_of_behaviour_plane)
            for st_i, st in enumerate(selected_traces_out_of_behaviour_plane):
                pca_st = pca_transformer.transform(st)
                axes[1, 2].plot(*st.T, color = 'green', alpha = 0.4, zorder = 100 + st_i)
            axes[1,2].scatter(novel_samples_out_of_behaviour_plane[:, 0], novel_samples_out_of_behaviour_plane[:, 1], alpha=0.5, s=1, c=network_input_classes)
            axes[1,2].set_title('novel samples orthogonal to behaviour plane')

            axes[0,2].set_title('Full noising of GT (out of behaviour space plane)')
            final_x_out_of_plane = (x_samples[:,-1] @ ddpm_model.linking_matrix_kernel.T).detach().cpu().numpy()
            pca_final_x = pca_transformer.transform(final_x_out_of_plane)
            axes[0,2].scatter(pca_final_x[:,0], pca_final_x[:,1], alpha=0.5, s=1, c=network_input_classes)

            symmetrize_and_square_axis(axes[0,2], 2.0)
            symmetrize_and_square_axis(axes[1,2], 2.0)

        elif ddpm_model.hidden_size == 3:
            raise NotImplementedError

        symmetrize_and_square_axis(axes[0,0], 2.0)
        symmetrize_and_square_axis(axes[1,0], 2.0)
        symmetrize_and_square_axis(axes[0,1], 2.0)
        symmetrize_and_square_axis(axes[1,1], 2.0)

        ## PLOT ALL LOSSES
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

        ## PLOT ALL LOSSES, ZOOMED INT
        axes[3,0].set_ylim(np.nanmin(all_total_losses[:t+1]), get_upper_ol_limit(all_total_losses[:t+1]))
        if num_hidden_steps > 1:
            axes[3,1].set_ylim(np.nanmin(all_individual_kl_losses[:t+1]), get_upper_ol_limit(all_individual_kl_losses[:t+1]))
        lossax2.set_ylim(np.nanmin(all_recon_losses[:t+1]), get_upper_ol_limit(all_recon_losses[:t+1]))

        cax = inset_axes(axes[2,1],
                    width="30%", # width = 30% of parent_bbox
                    height=1., # height : 1 inch
                    loc=3)
        plt.colorbar(kl_colors_scalarMap, cax = cax, ticks=range(1, num_hidden_steps, 10), orientation='vertical')

        plt.savefig(os.path.join(save_base, f"log.png"))
        plt.close('all')
        
        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.mdl"))


    elapsed_string, remaining_string = timer.loop_end()
    with open(training_print_path, 'a') as f:
        new_print_row = [
            f"{t + 1}/{num_trials}",                                                                 # Progress
            optim.param_groups[0]['lr'],
            round(all_recon_losses[t], 6),
        ] + [
            all_individual_kl_losses[t,i] for i in range(num_hidden_steps-1)
        ]
        new_print_row.extend([elapsed_string, remaining_string])
        print(*new_print_row, sep = '\t', file=f)

