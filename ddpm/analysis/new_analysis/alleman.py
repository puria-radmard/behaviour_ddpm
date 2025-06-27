import torch, sys, os
from analysis.new_analysis.utils import get_precue_elbo, get_normalised_projection_with_moments, get_postcue_elbo, display_normalised_projection_timeseries

from analysis.decoders import CuedIndexDependentReportFeatureProjector, ProbeFeatureDependentReportFeatureProjector

from ddpm.model import MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess

from tqdm import tqdm
import numpy as np

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.arguments_yaml import ConfigNamepace

from scipy.stats import ttest_rel

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1], strict_access = True)
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]
save_base = '/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/analysis/alleman_analysis'
save_path = os.path.join(save_base, yaml_name)

# run_name = 'run_b2_probe_cued_with_probe_flat_swap_fewer_variable_delay_0'
run_name = analysis_args.run_name

device = 'cuda'
_, task, ddpm_model, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_curriculum_activity_reg_20250322/{run_name}/args.yaml', device)
ddpm_model: MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
num_neurons = ddpm_model.sample_ambient_dim
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_curriculum_activity_reg_20250322/{run_name}/state.mdl', weights_only=True))

ddpm_model.eval()


stim_duration = task.task_variable_gen.prep_epoch_durations[0]
cuing_duration = task.task_variable_gen.prep_epoch_durations[2]

include_cuing_time = analysis_args.include_cuing_time
same_gating_across_time = analysis_args.same_gating_across_time
shared_p_misbind_over_time = analysis_args.shared_p_misbind_over_time
shared_p_swap_over_trials = analysis_args.shared_p_swap_over_trials
postcue_only_nullspace = analysis_args.postcue_only_nullspace
z_score = analysis_args.z_score
do_elbo = analysis_args.do_elbo
training_style = analysis_args.training_style
assert training_style in [
    'all_steps', 'first_step', 'last_step'
]

if not same_gating_across_time:
    assert training_style == 'all_steps'


ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps

cued_feature = analysis_args.cued_feature

first_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
precue_duration = first_delay_duration + stim_duration if include_cuing_time else first_delay_duration
diffusion_duration = ddpm_model.T + cuing_duration if include_cuing_time else ddpm_model.T

if cued_feature == 'index':
    mode_generator = CuedIndexDependentReportFeatureProjector(
        dim_K = 8, 
        dim_R = num_neurons,
        precue_duration = precue_duration,
        postcue_duration = diffusion_duration,
        same_gating_across_time = same_gating_across_time,
        shared_p_misbind_over_time = shared_p_misbind_over_time,
    )
elif cued_feature == 'probe':
    mode_generator = ProbeFeatureDependentReportFeatureProjector(
        dim_K = 8,
        dim_R = num_neurons,
        precue_duration = precue_duration,
        postcue_duration = diffusion_duration,
        same_gating_across_time = same_gating_across_time,
        shared_p_misbind_over_time = shared_p_misbind_over_time
    )
else:
    raise ValueError(cued_feature)
mode_generator.cuda()

all_parameters = list(mode_generator.parameters())

optim = torch.optim.Adam(all_parameters, lr = 0.1)


precue_cmap = cmx.ScalarMappable(norm=colors.Normalize(vmin=1, vmax=precue_duration), cmap=plt.get_cmap("viridis"))
precue_cmap.set_array([])

diffusion_cmap = cmx.ScalarMappable(norm=colors.Normalize(vmin=1, vmax=diffusion_duration), cmap=plt.get_cmap("magma"))
diffusion_cmap.set_array([])


_, save_path, _ = configure_logging_paths(save_path, [])
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))
print(save_path)


# Only accept trials where p_correct > 0.7 or < 0.3
p_decision_thres = 0.6


all_precue_positional_decoder_elbos = np.zeros([num_opt_steps, mode_generator.precue_duration])
all_postcue_positional_decoder_elbos = np.zeros([num_opt_steps, mode_generator.postcue_duration])

infinite_data_mode = True
num_samples = 1


#print('RESUMING MODEL FROM results_link_sampler/analysis/alleman_analysis/index_21/mode_generator.mdl')
#mode_generator.load_state_dict(torch.load('results_link_sampler/analysis/alleman_analysis/index_21/mode_generator.mdl', weights_only=True))


for i in tqdm(range(num_opt_steps)):

    if infinite_data_mode or i == 0:

        trial_information = task.generate_trial_information(batch_size=ctd_batch_size, num_samples=num_samples)

        trial_information.task_variable_information['prep_epoch_durations'][1] = first_delay_duration

        with torch.no_grad():
            novel_samples_prep_dicts, novel_samples_dict = ddpm_model.generate_samples(
                prep_network_inputs = trial_information.prep_network_inputs,
                diffusion_network_inputs = trial_information.diffusion_network_inputs,
                prep_epoch_durations=trial_information.prep_epoch_durations,
                diffusion_epoch_durations=trial_information.diffusion_epoch_durations,
                samples_shape=[ctd_batch_size, num_samples],
                noise_scaler=1.0,
            )

            # Determine whether swap or correct has p > 0.7, under Gaussian likelihood on 2D plane
            square_cart_errors = ((novel_samples_dict['samples'] / task.sample_gen.sample_radius) - trial_information.task_variable_information['report_features_cart']).square().sum(-1)    # [B, N]
            p_components = (- square_cart_errors).softmax(-1)
            selected_trials_mask = (p_components >= p_decision_thres).any(-1)

            remaining_p_corrects = p_components[selected_trials_mask][
                torch.arange(selected_trials_mask.sum()), 
                trial_information.task_variable_information['cued_item_idx'][selected_trials_mask]
            ].unsqueeze(1).cuda()
            correct_trials_mask = remaining_p_corrects.squeeze(1) >= p_decision_thres

            p_correct = (remaining_p_corrects).mean() if shared_p_swap_over_trials else (1.0 - remaining_p_corrects)

            first_delay_activity = novel_samples_prep_dicts[1]['preparatory_trajectory'][selected_trials_mask].squeeze(1)  # [batch dim, duration1]


            diffusion_time_activity = novel_samples_dict['embedded_sample_trajectory'][selected_trials_mask].cuda().squeeze(1)  # [batch dim, duration2]

            if include_cuing_time:
                stim_activity = novel_samples_prep_dicts[0]['preparatory_trajectory'][selected_trials_mask].squeeze(1)  # [batch dim, duration1]
                first_delay_activity = torch.concat([stim_activity, first_delay_activity], dim = 1)

                cue_activity = novel_samples_prep_dicts[2]['preparatory_trajectory'][selected_trials_mask].squeeze(1)  # [batch dim, duration1]
                diffusion_time_activity = torch.concat([cue_activity, diffusion_time_activity], dim = 1)

            if postcue_only_nullspace:
                diffusion_time_activity = diffusion_time_activity @ ddpm_model.behaviour_nullspace_accessor.T


        if z_score:
            elementwise_first_delay_stds = first_delay_activity[:,-first_delay_duration].std(0)[None,None]
            elementwise_diffusion_stds = diffusion_time_activity[:,-ddpm_model.T].std(0)[None,None]
            elementwise_first_delay_means = first_delay_activity[:,-first_delay_duration].mean(0)[None,None]
            elementwise_diffusion_means = diffusion_time_activity[:,-ddpm_model.T].mean(0)[None,None]


    if z_score:
        first_delay_activity = (first_delay_activity - elementwise_first_delay_means) / elementwise_first_delay_stds
        diffusion_time_activity = (diffusion_time_activity - elementwise_diffusion_means) / elementwise_diffusion_stds

    if i == 0:
        # print('NOT USING DATA VARIANCE')
        mode_generator.mode_variance_raw.data = first_delay_activity.var(1).mean(0).log()
        
        
    report_feature = trial_information.task_variable_information['report_features_cart'][selected_trials_mask].cuda().to(first_delay_activity.dtype)
    cued_indices = trial_information.task_variable_information['cued_item_idx'][selected_trials_mask].cuda().to(first_delay_activity.dtype)
    probe_features = trial_information.task_variable_information['report_features_cart'][selected_trials_mask].cuda().to(first_delay_activity.dtype)

    probing_features = cued_indices if cued_feature == 'index' else probe_features

    total_precue_elbo, nominal_precue_modes, misbound_modes = get_precue_elbo(
        probing_features, report_feature, mode_generator, first_delay_activity, p_correct, do_elbo
    )
    total_postcue_elbo, nominal_postcue_modes, misselected_modes, misinterpreted_modes = get_postcue_elbo(
        probing_features, report_feature, mode_generator, diffusion_time_activity, cued_indices, p_correct, do_elbo
    )

    # raise Exception('decide which parts to train on!')
    optim.zero_grad()
    if training_style == 'all_steps':
        loss = - total_precue_elbo[-first_delay_duration:].sum() - total_postcue_elbo[-ddpm_model.T:].sum()
    elif training_style == 'first_step':
        loss = - total_precue_elbo[-first_delay_duration].sum() - total_postcue_elbo[-ddpm_model.T].sum()
    loss.backward()
    optim.step()

    all_precue_positional_decoder_elbos[i] = total_precue_elbo.tolist()
    all_postcue_positional_decoder_elbos[i] = total_postcue_elbo.tolist()


    if i % 50 == 0:

        plt.close('all')

        correct_trials_precue_normalised_projection, swap_trials_precue_normalised_projection = get_normalised_projection_with_moments(
            nominal_precue_modes, misbound_modes, first_delay_activity, correct_trials_mask
        )

        correct_trials_postcue_misselected_normalised_projection, swap_trials_postcue_misselected_normalised_projection = get_normalised_projection_with_moments(
            nominal_postcue_modes, misselected_modes, diffusion_time_activity, correct_trials_mask
        )

        correct_trials_postcue_misinterpreted_normalised_projection, swap_trials_postcue_misinterpreted_normalised_projection = get_normalised_projection_with_moments(
            nominal_postcue_modes, misinterpreted_modes, diffusion_time_activity, correct_trials_mask
        )


        ### PLOT TRAINING LOSSES
        fig, axes = plt.subplot_mosaic(
            """
            ABCDEF
            GHIJKL
            MNOPQR
            XXXSTU
            """, figsize = (30, 15)
        )
        
        
        for j in range(mode_generator.precue_duration):
            color = precue_cmap.to_rgba(j + 1)

            zoomed_positional_decoder_losses_line = all_precue_positional_decoder_elbos[i+1-50:i+1,j]
            positional_decoder_losses_line = all_precue_positional_decoder_elbos[:i+1,j]

            axes['A'].plot(zoomed_positional_decoder_losses_line, color = color)
            axes['B'].plot(positional_decoder_losses_line, color = color)
            
        axes['A'].set_title('zoomed precue elbo over training')
        axes['B'].set_title('precue elbo over training')

        axes['C'].plot(all_precue_positional_decoder_elbos[i])
        axes['C'].set_title('recent precue elbo over delay')

        axes['H'].plot(mode_generator.p_misbind.data.detach().cpu().numpy()[0])
        axes['H'].set_title('p_misbind over delay')

        axes['G'].plot(sorted(mode_generator.mode_variance_raw.exp().data.tolist()))
        axes['G'].set_title('variance over dimensions')

        for j in range(mode_generator.postcue_duration):
            color = diffusion_cmap.to_rgba(j + 1)

            zoomed_positional_decoder_losses_line = all_postcue_positional_decoder_elbos[i+1-50:i+1,j]
            positional_decoder_losses_line = all_postcue_positional_decoder_elbos[:i+1,j]

            axes['D'].plot(zoomed_positional_decoder_losses_line, color = color)
            axes['E'].plot(positional_decoder_losses_line, color = color)
            
        criterion_name = 'elbo' if do_elbo else 'likelihood'
        axes['D'].set_title(f'zoomed postcue {criterion_name} over training')
        axes['E'].set_title(f'postcue {criterion_name} over training')

        axes['F'].plot(all_postcue_positional_decoder_elbos[i])
        axes['F'].set_title(f'recent precue {criterion_name} over diffusion')

        _, p_misselect, p_misinterpret = mode_generator.p_post_cue_errors.data.detach().cpu().numpy()
        axes['K'].plot(p_misselect[0])
        axes['L'].plot(p_misinterpret[0])
        axes['K'].set_title('p_misselect over delay')
        axes['L'].set_title('p_misinterpret over delay')
        
        axes['J'].plot(sorted(mode_generator.mode_variance_raw.exp().data.tolist()))
        axes['J'].set_title('variance over dimensions CHANGE')     

        display_normalised_projection_timeseries(correct_trials_precue_normalised_projection, swap_trials_precue_normalised_projection, [axes['M'], axes['N'], axes['O']], 'purple')
        display_normalised_projection_timeseries(correct_trials_postcue_misselected_normalised_projection, swap_trials_postcue_misselected_normalised_projection, [axes['P'], axes['Q'], axes['R']], 'pink')
        display_normalised_projection_timeseries(correct_trials_postcue_misinterpreted_normalised_projection, swap_trials_postcue_misinterpreted_normalised_projection, [axes['S'], axes['T'], axes['U']], 'orange')

        axes['M'].set_ylabel('Normalised projection - misbind')
        axes['P'].set_ylabel('Normalised projection - misselect')
        axes['S'].set_ylabel('Normalised projection - misinterpret')


        if include_cuing_time:
            for key in 'MNO':
                ylim = axes[key].get_ylim()
                axes[key].plot([stim_duration, stim_duration], ylim, linestyle = '--', color = 'black')
                axes[key].set_ylim(ylim)

            for key in 'PQRSTU':
                ylim = axes[key].get_ylim()
                axes[key].plot([cuing_duration, cuing_duration], ylim, linestyle = '--', color = 'black')
                axes[key].set_ylim(ylim)


        fig.savefig(os.path.join(save_path, 'loss'))
        
        torch.save(mode_generator.state_dict(), os.path.join(save_path, 'mode_generator.mdl'))
