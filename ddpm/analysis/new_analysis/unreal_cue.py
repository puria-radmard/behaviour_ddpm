

from matplotlib import pyplot as plt, colors, cm as cmx

import torch, sys, os
import numpy as np

from purias_utils.util.arguments_yaml import ConfigNamepace

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch


from ddpm.tasks.main.multiepoch import MultiepochTrialInformation

from purias_utils.util.logging import configure_logging_paths

base_dir_name = 'hvae_with_dendritic_20250410'


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1], strict_access = True)
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]
save_base = '/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/analysis/unreal_cue'
save_path = os.path.join(save_base, yaml_name)
print(save_path)

_, save_path, _ = configure_logging_paths(save_path, [])
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))


print(save_path)


run_name = analysis_args.run_name


device = 'cuda'
_, task, ddpm_model, _, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/{base_dir_name}/{run_name}/args.yaml', device)
num_neurons = ddpm_model.sample_ambient_dims[-1]
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/{base_dir_name}/{run_name}/state.mdl', weights_only=True))



num_cue_gridpoints = 100
num_samples = 100

fixed_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])


batch_size = 2 + num_cue_gridpoints       # These will have the same report, only the cue will be gridded amongst them!


fixed_probe_locations = torch.tensor([- torch.pi/2, + torch.pi/2])
fixed_report_locations = torch.tensor([0, + torch.pi/2])

# assert (fixed_report_locations[1] - fixed_report_locations[0]).abs() > torch.pi / 2, "Using swap probability heuristic!"

cues_to_try = torch.linspace(-torch.pi, +torch.pi, num_cue_gridpoints + 1)[:-1]
cues_to_try = torch.concat([fixed_probe_locations, cues_to_try])

fixed_probe_locations_cart = torch.stack([fixed_probe_locations.cos(), fixed_probe_locations.sin()], -1)
fixed_report_locations_cart = torch.stack([fixed_report_locations.cos(), fixed_report_locations.sin()], -1)
cues_to_try_cart = torch.stack([cues_to_try.cos(), cues_to_try.sin()], -1)

task_variable_information = task.task_variable_gen.generate_variable_dict(batch_size=batch_size)

task_variable_information['probe_features'] = fixed_probe_locations[None].expand(batch_size, 2)
task_variable_information['report_features'] = fixed_report_locations[None].expand(batch_size, 2)
task_variable_information['probe_features_cart'] = fixed_probe_locations_cart[None].expand(batch_size, 2, 2)   # Replace all probe features
task_variable_information['report_features_cart'] = fixed_report_locations_cart[None].expand(batch_size, 2, 2)   # Replace all report features
task_variable_information['override_cue_features_cart'] = cues_to_try_cart
task_variable_information['override_cue_features'] = cues_to_try.unsqueeze(1)



prep_network_inputs = task.sensory_gen.generate_prep_sensory_inputs(task_variable_information)
diffusion_network_inputs = task.sensory_gen.generate_diffusion_sensory_inputs(task_variable_information)
prep_network_inputs = [
    pni.unsqueeze(1).expand(pni.shape[0], num_samples, *pni.shape[1:])
    for pni in prep_network_inputs
]
diffusion_network_inputs = [
    dni.unsqueeze(1).expand(dni.shape[0], num_samples, *dni.shape[1:])
    for dni in diffusion_network_inputs
]
sample_information = task.sample_gen.generate_sample_set(num_samples, task_variable_information)
prep_epoch_durations = task_variable_information["prep_epoch_durations"]
diffusion_epoch_durations = task_variable_information["diffusion_epoch_durations"]
trial_information = MultiepochTrialInformation(
    task_variable_information,
    prep_network_inputs,
    diffusion_network_inputs,
    prep_epoch_durations,
    diffusion_epoch_durations,
    sample_information,
)


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
        samples_shape=[batch_size, num_samples],
        noise_scaler=1.0,
    )

final_samples_cart = novel_samples_dict['samples']  # [B, S, 2]
final_samples = torch.arctan2(final_samples_cart[...,1], final_samples_cart[...,0]) # [B, S]
final_samples = final_samples.cpu().numpy()


fig, axes = plt.subplots(2, 2, figsize = (12, 12))
axes = axes.flatten()
all_hists = []

all_hists.append(axes[0].hist(final_samples[0], label = 'Real probe 1', bins = 64, fc='none', density = False, histtype='step', color = 'blue'))
all_hists.append(axes[0].hist(final_samples[1], label = 'Real probe 2', bins = 64, fc='none', density = False, histtype='step', color = 'green'))

magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=num_cue_gridpoints)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])
for j in range(num_cue_gridpoints):
    color = kl_colors_scalarMap.to_rgba(j + 1)
    all_hists.append(axes[0].hist(final_samples[j+2], bins = 64, fc='none', density = False, histtype='step', color = color))
    


from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

print("Using swap probability heuristic!")

modal_bins = [hi[0].argmax() for hi in all_hists]
bin_midpoints = [0.5 * (hi[1][1:] + hi[1][:-1]) for hi in all_hists]
modal_estimates = [bmp[mb] for mb, bmp in zip(modal_bins, bin_midpoints)]
distance_to_modes = [np.abs(rectify_angles(mp - mb)) for mp, mb in zip(bin_midpoints, modal_estimates)]
is_swap_masks = [dtm > torch.pi / 3 for dtm in distance_to_modes]
swap_probabilities = [hi[0][ism].sum() / num_samples for hi, ism in zip(all_hists, is_swap_masks)]

try:
    if task.sensory_gen.cue_with_report:
        axes[1].plot([-torch.pi, torch.pi], [-torch.pi, torch.pi], color = 'black', linestyle = '--')
except AttributeError:
    pass


axes[1].set_xlabel('Cued feature value')
axes[1].set_ylabel('Modal estimate recall')

axes[2].set_xlabel('Cued feature value')
axes[2].set_ylabel('Swap probability')

for j in range(batch_size):
    
    if j == 0:
        color = 'blue'
    elif j == 1:
        color = 'green'
    else:
        color = kl_colors_scalarMap.to_rgba(j - 1)
    
    axes[1].scatter([cues_to_try[j]], [modal_estimates[j]], color = color, marker = 'x')

    axes[2].scatter([cues_to_try[j]], [swap_probabilities[j]], color = color, marker = 'x')

fig.savefig(os.path.join(save_path, 'different_cues'))
