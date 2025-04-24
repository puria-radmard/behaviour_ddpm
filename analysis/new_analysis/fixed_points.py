import torch, sys, os

from tqdm import tqdm
from torch import Tensor as _T

import matplotlib.pyplot as plt
from ddpm.model.main.multiepoch import MultiPreparatoryHVAEReverseProcess

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles, polar2cart

from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.util.logging import configure_logging_paths

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


base_dir_name = 'hvae_with_dendritic_20250410'

device = 'cuda'


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1], strict_access = True)
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]
save_base = '/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/analysis/fixed_points'
save_path = os.path.join(save_base, yaml_name)

_, save_path, _ = configure_logging_paths(save_path, [])
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))


print(save_path)


run_name = analysis_args.run_name


_, task, ddpm_model, _, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/{base_dir_name}/{run_name}/args.yaml', device)
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/{base_dir_name}/{run_name}/state.mdl', weights_only=True))
ddpm_model: MultiPreparatoryHVAEReverseProcess

for n, parameter in ddpm_model.named_parameters():
    parameter.required_grad = False

minimum_margin = torch.pi / task.task_variable_gen.min_sep
grid_points = 12
stim1_probe_features = torch.linspace(0, 2 * torch.pi, grid_points+1)[:-1].unsqueeze(1).repeat(1, grid_points)
stim2_probe_features = rectify_angles(torch.linspace(minimum_margin, 2 * torch.pi - minimum_margin, grid_points+1)[:-1].unsqueeze(0) + stim1_probe_features[:,[0]])
all_probe_features = torch.stack([rectify_angles(stim1_probe_features), stim2_probe_features], -1).reshape(grid_points * grid_points, 2)

all_report_features = all_probe_features.clone()

all_probe_features = all_probe_features.unsqueeze(1).repeat(1, grid_points * grid_points, 1).reshape(grid_points ** 4, 2)
all_report_features = all_report_features.unsqueeze(0).repeat(grid_points * grid_points, 1, 1).reshape(grid_points ** 4, 2)


all_only_stim2_changes_mask = 

unique_probe_combinations = all_probe_features[::grid_points*grid_points]



both_like_first_probe_mask = (all_probe_features == all_probe_features[[0]]).all(-1)
stim1_like_first_report_mask = (all_report_features[:,[0]] == all_report_features[[0]][:,[0]]).squeeze(-1)
only_stim2_changes_mask = both_like_first_probe_mask & stim1_like_first_report_mask

import pdb; pdb.set_trace()



batch_size = grid_points ** 4
num_samples = 1
color_array = torch.arange(batch_size).unsqueeze(1).repeat(1, num_samples)
color_array_flattened = color_array.reshape(batch_size * num_samples)


override_stimulus_features = {'probe_features': all_probe_features, 'report_features': all_report_features}
override_stimulus_cart_features = {}
for k in override_stimulus_features.keys():
    override_stimulus_cart_features[f'{k}_cart'] = torch.stack(polar2cart(1.0, override_stimulus_features[k]), -1)

override_stimulus_features_dict = dict(**override_stimulus_features, **override_stimulus_cart_features)

task_variable_dict = task.task_variable_gen.generate_variable_dict(
    batch_size = batch_size,
    override_stimulus_features_dict=override_stimulus_features_dict
)

trial_information = task.generate_trial_information(
    batch_size=batch_size, num_samples=num_samples, override_task_variable_information = task_variable_dict
)

all_feature_data = torch.concat([
    trial_information.task_variable_information['probe_features_cart'].reshape(batch_size, 4),
    trial_information.task_variable_information['report_features_cart'].reshape(batch_size, 4),
], dim = -1)


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

prep_epoch_idx = 1


delay_activity = novel_samples_prep_dicts[prep_epoch_idx]['preparatory_trajectory']  # [batch dim, samples, duration1, neurons]
delay_duration = delay_activity.shape[2]


def energy_function(neural_response: _T):
    delay_input = trial_information.prep_network_inputs[prep_epoch_idx]
    assert (delay_input == 0.0).all()
    time_embedding = ddpm_model.prep_time_embedding.expand(delay_duration, ddpm_model.prep_time_embedding.shape[1])
    network_input = ddpm_model.input_model(delay_input, delay_duration)
    xdot: _T = - neural_response + ddpm_model.residual_model(neural_response, time_embedding, network_input)
    return 0.5 * xdot.square().sum()


num_fp_opt_steps = 100000
activity_patterns = torch.nn.Parameter(delay_activity, requires_grad = True)
optim = torch.optim.Adam([activity_patterns], lr = 1e-2)

all_energies = []

for ots in tqdm(range(num_fp_opt_steps)):

    energy = energy_function(activity_patterns)
    total_energy = energy.mean()

    optim.zero_grad()
    total_energy.backward()
    optim.step()

    all_energies.append(total_energy.item())

    if ots % 100 == 0:

        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize = (20, 20))

        # Visualise PCA
        pcaable_data = activity_patterns.data[:,:,0,:].cpu().reshape(batch_size * num_samples, -1) # only one timestep rn
        pca = PCA(n_components=2, whiten = True).fit(pcaable_data)
        axes[0,1].scatter(*pca.transform(pcaable_data).T, c = color_array_flattened)

        # Visualise selected examples
        selected_pcaable_data = pcaable_data[only_stim2_changes_mask]
        pca_masked = PCA(n_components=2, whiten = True).fit(selected_pcaable_data)
        axes[1,0].scatter(*pca_masked.transform(selected_pcaable_data).T)

        # # Visualise CCA
        # item1_features = all_feature_data[:,[0,1,4,5]]
        # cca = CCA(n_components=2).fit(pcaable_data, item1_features)
        # activity_cross, stim_cross = cca.transform(pcaable_data, item1_features)
        # axes[1,0].scatter(*activity_cross.T, c = color_array_flattened)
        
        # Loss
        axes[0,0].plot(all_energies)
        
        fig.savefig('analysis/new_analysis/fixed_points_test.png')
