from analysis.decoder_analysis.shared_setup import *

from ddpm.tasks.main.multiepoch import MultiepochTrialInformation


first_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
diffusion_duration = ddpm_model.T       # Not used here though!


magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=first_delay_duration)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])

include_cuing_time = True
cuing_duration = task.task_variable_gen.prep_epoch_durations[0]

same_gating_across_time = False
shared_p_misbind_over_time = True

cued_feature = analysis_args.cued_feature


precue_duration = first_delay_duration + cuing_duration if include_cuing_time else first_delay_duration

if cued_feature == 'index':
    mode_generator = CuedIndexDependentReportFeatureProjector(
        dim_K = 8, 
        dim_R = num_neurons,
        precue_duration = precue_duration,
        postcue_duration = diffusion_duration,
        same_gating_across_time = same_gating_across_time,
        shared_p_misbind_over_time = shared_p_misbind_over_time
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

optim = torch.optim.Adam(all_parameters, lr = 0.01)



resumed = False

save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')
    try:
        mode_generator.load_state_dict(torch.load(os.path.join(save_path, 'mode_generator.mdl')))
        resumed = True
        # print('NOT RESUMING FROM FILE')
        pass
    except (FileNotFoundError, RuntimeError):
        print('NO SAVED MODEL TO RESUME FROM')
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))




ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps


# Only accept trials where p_correct > 0.7 or < 0.3
p_decision_thres = 0.6


all_positional_decoder_losses = np.zeros([num_opt_steps, mode_generator.precue_duration])

infinite_data_mode = True
num_samples = 1024
ctd_batch_size = 1




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

    no_noise_novel_samples_prep_dicts, no_noise_novel_samples_dict = ddpm_model.generate_samples(
        prep_network_inputs = trial_information.prep_network_inputs,
        diffusion_network_inputs = trial_information.diffusion_network_inputs,
        prep_epoch_durations=trial_information.prep_epoch_durations,
        diffusion_epoch_durations=trial_information.diffusion_epoch_durations,
        samples_shape=[ctd_batch_size, num_samples],
        noise_scaler=0.0,
    )

trial_information.task_variable_information['cued_item_idx'] = 1 - trial_information.task_variable_information['cued_item_idx']

prep_network_inputs = task.sensory_gen.generate_prep_sensory_inputs(
    trial_information.task_variable_information
)
diffusion_network_inputs = task.sensory_gen.generate_diffusion_sensory_inputs(
    trial_information.task_variable_information
)
prep_network_inputs = [
    pni.unsqueeze(1).expand(pni.shape[0], num_samples, *pni.shape[1:])
    for pni in prep_network_inputs
]
diffusion_network_inputs = [
    dni.unsqueeze(1).expand(dni.shape[0], num_samples, *dni.shape[1:])
    for dni in diffusion_network_inputs
]
sample_information = task.sample_gen.generate_sample_set(
    num_samples, trial_information.task_variable_information
)
prep_epoch_durations = trial_information.task_variable_information["prep_epoch_durations"]
diffusion_epoch_durations = trial_information.task_variable_information["diffusion_epoch_durations"]
swapped_trial_information = MultiepochTrialInformation(
    trial_information.task_variable_information,
    prep_network_inputs,
    diffusion_network_inputs,
    prep_epoch_durations,
    diffusion_epoch_durations,
    sample_information,
)


with torch.no_grad():
    swapped_novel_samples_prep_dicts, swapped_novel_samples_dict = ddpm_model.generate_samples(
        prep_network_inputs = swapped_trial_information.prep_network_inputs,
        diffusion_network_inputs = swapped_trial_information.diffusion_network_inputs,
        prep_epoch_durations=swapped_trial_information.prep_epoch_durations,
        diffusion_epoch_durations=swapped_trial_information.diffusion_epoch_durations,
        samples_shape=[ctd_batch_size, num_samples],
        noise_scaler=1.0,
    )

    no_noise_swapped_novel_samples_prep_dicts, no_noise_swapped_novel_samples_dict = ddpm_model.generate_samples(
        prep_network_inputs = swapped_trial_information.prep_network_inputs,
        diffusion_network_inputs = swapped_trial_information.diffusion_network_inputs,
        prep_epoch_durations=swapped_trial_information.prep_epoch_durations,
        diffusion_epoch_durations=swapped_trial_information.diffusion_epoch_durations,
        samples_shape=[ctd_batch_size, num_samples],
        noise_scaler=0.0,
    )


fig, axes = plt.subplots(2, 2, figsize = (10, 10))

axes[0,0].set_title('Original cue')
task.sample_gen.display_samples(novel_samples_dict["samples"], axes[0,0])

axes[0,1].set_title('Swapped cue')
task.sample_gen.display_samples(swapped_novel_samples_dict["samples"], axes[0,1])

axes[1,0].set_title('No noise, original cue')
task.sample_gen.display_samples(no_noise_novel_samples_dict["samples"], axes[1,0])

axes[1,1].set_title('No noise, swapped cue')
task.sample_gen.display_samples(no_noise_swapped_novel_samples_dict["samples"], axes[1,1])

fig.savefig(os.path.join(save_path, 'cue_swap_check'))
