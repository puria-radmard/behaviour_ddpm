from analysis.decoder_analysis.shared_setup import *
from ddpm.tasks.main.base import TrialInformation


save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))


num_samples = analysis_args.num_samples

try:
    fixed_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
except TypeError:
    fixed_delay_duration = task.task_variable_gen.prep_epoch_durations[1]


batch_size = 2 + analysis_args.num_cue_gridpoints       # These will have the same report, only the cue will be gridded amongst them!


fixed_probe_locations = torch.tensor([-2 *torch.pi / 3, torch.pi / 3])
fixed_report_locations = torch.tensor([+torch.pi / 2, -2 * torch.pi / 2])

assert (fixed_report_locations[1] - fixed_report_locations[0]).abs() > torch.pi / 2, "Using swap probability heuristic!"

cues_to_try = torch.linspace(-torch.pi, +torch.pi, analysis_args.num_cue_gridpoints + 1)[:-1]
cues_to_try = torch.concat([fixed_probe_locations, cues_to_try])

fixed_probe_locations_cart = torch.stack([fixed_probe_locations.cos(), fixed_probe_locations.sin()], -1)
fixed_report_locations_cart = torch.stack([fixed_report_locations.cos(), fixed_report_locations.sin()], -1)
cues_to_try_cart = torch.stack([cues_to_try.cos(), cues_to_try.sin()], -1)

task_variable_information = task.task_variable_gen.generate_variable_dict(batch_size=batch_size)

task_variable_information.pop('probe_features')
task_variable_information.pop('report_features')
task_variable_information['probe_features_cart'] = fixed_probe_locations_cart[None].expand(batch_size, 2, 2)   # Replace all probe features
task_variable_information['report_features_cart'] = fixed_report_locations_cart[None].expand(batch_size, 2, 2)   # Replace all report features
task_variable_information['override_cue_features_cart'] = cues_to_try_cart


network_inputs = task.sensory_gen.generate_sensory_inputs(
    task_variable_information
)
network_inputs = network_inputs.unsqueeze(1).expand(
    batch_size, num_samples, *task.sensory_gen.sensory_shape
)
sample_information = task.sample_gen.generate_sample_set(
    num_samples, task_variable_information
)
trial_information = TrialInformation(
    task_variable_information, network_inputs, sample_information
)


trial_information.task_variable_information['prep_epoch_durations'][1] = fixed_delay_duration

with torch.no_grad():
    novel_samples_dict = ddpm_model.generate_samples(
        network_input = trial_information.network_inputs,
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
cNorm = colors.Normalize(vmin=1, vmax=analysis_args.num_cue_gridpoints)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])
for j in range(analysis_args.num_cue_gridpoints):
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

# if task.sensory_gen.cue_with_report:
#     axes[1].plot([-torch.pi, torch.pi], [-torch.pi, torch.pi], color = 'black', linestyle = '--')


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
