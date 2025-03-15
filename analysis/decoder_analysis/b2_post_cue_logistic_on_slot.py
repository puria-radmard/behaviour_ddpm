import torch, sys, os

from torch import Tensor as _T

from analysis.decoder_analysis.decoders import *

from tqdm import tqdm
import numpy as np

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from purias_utils.util.arguments_yaml import ConfigNamepace

from scipy.stats import ttest_rel

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1])
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]


# run_name = 'run_b2_probe_cued_with_probe_flat_swap_fewer_variable_delay_0'
run_name = analysis_args.run_name

device = 'cuda'
_, task, ddpm_model, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_further_20250120/{run_name}/args.yaml', device)
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_further_20250120/{run_name}/state.mdl'))

ddpm_model.eval()

num_neurons = ddpm_model.sample_ambient_dim


if analysis_args.delay_period:
    raise Exception('This makes no sense')
    fixed_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
    relevant_duration = fixed_delay_duration
else:
    relevant_duration = ddpm_model.sigma2xt_schedule.shape[0]
    fixed_delay_duration = 20   # XXX remove hard coding!


decode_cued = analysis_args.decode_cued
limit_to_further = analysis_args.limit_to_further


save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')

analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))

ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps

decoding_variable = 'report_features_cart'

decoder_type = analysis_args.decoder_type

decoder = CrossTemporalLinearDecoder(relevant_duration, num_neurons, output_size=1)

optim = torch.optim.Adam(decoder.parameters())


all_positional_decoder_losses = np.zeros([num_opt_steps, relevant_duration])
all_other_positional_decoder_losses = np.zeros([num_opt_steps, relevant_duration])

num_samples = 1

infinite_data_mode = True

magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=relevant_duration)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])

for i in tqdm(range(num_opt_steps)):

    if infinite_data_mode or i == 0:

        trial_information = task.generate_trial_information(batch_size=ctd_batch_size, num_samples=num_samples)

        trial_information.task_variable_information['prep_epoch_durations'][1] = fixed_delay_duration

        with torch.no_grad():
            novel_samples_prep_dicts, novel_samples_dict = ddpm_model.generate_samples(
                # prep_network_inputs=[
                #     pni[[0]] for pni in trial_information.prep_network_inputs
                # ],
                # diffusion_network_inputs=[
                #     dni[[0]] for dni in trial_information.diffusion_network_inputs
                # ],
                prep_network_inputs = trial_information.prep_network_inputs,
                diffusion_network_inputs = trial_information.diffusion_network_inputs,
                prep_epoch_durations=trial_information.prep_epoch_durations,
                diffusion_epoch_durations=trial_information.diffusion_epoch_durations,
                samples_shape=[ctd_batch_size, num_samples],
                noise_scaler=1.0,
            )

        if analysis_args.delay_period:
            relevant_activity = novel_samples_prep_dicts[1]['preparatory_trajectory'].squeeze(1)  # [batch, time (30), dim (128)]
        else:
            relevant_activity = novel_samples_dict['embedded_sample_trajectory'].squeeze(1).cuda()  # [batch, time (30), dim (128)]

    cued_idxes: _T = trial_information.task_variable_information['cued_item_idx'].int()
    cued_idxes = cued_idxes[None,:,None].cuda().repeat(decoder.num_models, 1, relevant_duration)

    all_items = torch.arange(ctd_batch_size)
    output = decoder.decode_sequence(relevant_activity).squeeze(-1)  # [models, batch, time, 1]
    output_prob = torch.sigmoid(output)
    eps = 1e-05
    output_prob = output_prob + eps
    output_prob = output_prob / (1 + 2 * eps)


    loss = (- (output_prob.log() * cued_idxes) - ((1.0 - output_prob).log() * (1 - cued_idxes))).mean(1)
    

    optim.zero_grad()
    loss.mean(0).sum().backward()
    optim.step()

    all_positional_decoder_losses[i] = loss.mean(0).tolist()

    if i % 50 == 0:

        plt.close('all')


        ### PLOT TRAINING LOSSES
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))

        for j in range(relevant_duration):
            color = kl_colors_scalarMap.to_rgba(j + 1)

            zoomed_positional_decoder_losses_line = all_positional_decoder_losses[i+1-50:i+1,j]
            zoomed_other_positional_decoder_losses_line = all_other_positional_decoder_losses[i+1-50:i+1,j]
            positional_decoder_losses_line = all_positional_decoder_losses[:i+1,j]
            other_positional_decoder_losses_line = all_other_positional_decoder_losses[:i+1,j]

            axes[0,0].plot(zoomed_positional_decoder_losses_line, color = color)
            axes[0,1].plot(zoomed_other_positional_decoder_losses_line, color = color)
            axes[1,0].plot(positional_decoder_losses_line, color = color)
            axes[1,1].plot(other_positional_decoder_losses_line, color = color)
            
            axes[0,0].set_title('zoomed_positional_decoder_losses_line')
            axes[0,1].set_title('zoomed_other_positional_decoder_losses_line')
            axes[1,0].set_title('positional_decoder_losses_line')
            axes[1,1].set_title('other_positional_decoder_losses_line')

        fig.savefig(os.path.join(save_path, 'loss'))

        
        
        ### PLOT LOSS OVER TIME
        fig,axes = plt.subplots(1, figsize = (10, 6))

        correct_probs = output_prob.detach().cpu()
        cued_idxes = cued_idxes.cpu()
        correct_probs[cued_idxes == 0.0] = 1.0 - correct_probs[cued_idxes == 0.0]

        for j in range(relevant_duration):
            color = kl_colors_scalarMap.to_rgba(j + 1)

            correct_probs_at_ts = correct_probs[...,j]
            
            loss_x = (torch.ones(*correct_probs_at_ts.shape) * j)
            loss_x = (loss_x + (torch.randn_like(loss_x) * 0.2)).cpu().numpy()

            axes.scatter(loss_x, correct_probs_at_ts, color = color)

        axes.plot(axes.get_xlim(), [0, 0], color = 'black')
        fig.savefig(os.path.join(save_path, 'swapped_loss'))

        
        torch.save(ddpm_model.state_dict(), os.path.join(save_path, 'decoder.mdl'))

        np.save(os.path.join(save_path, 'recent_losses.mdl'), {'correct_probs': correct_probs.detach().cpu().numpy()})
