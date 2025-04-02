import torch, sys, os

from tqdm import tqdm

import matplotlib.pyplot as plt

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch

from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.plotting import standard_swap_model_simplex_plots
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole

# base_dir_name = 'ddpm_curriculum_activity_reg_20250322'
# base_dir_name = 'hvae_direct_param_20250327'
base_dir_name = 'hvae_with_bounceback_20250330'


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1], strict_access = True)
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]
save_base = '/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/analysis/bns_fits'
save_path = os.path.join(save_base, yaml_name)

_, save_path, _ = configure_logging_paths(save_path, [])
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))


run_name = analysis_args.run_name


device = 'cuda'
_, task, ddpm_model, _, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/{base_dir_name}/{run_name}/args.yaml', device)
num_neurons = ddpm_model.sample_ambient_dims[-1]
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/{base_dir_name}/{run_name}/state.mdl'))

ddpm_model.eval()



device = 'cuda'

emission_type = 'von_mises'
kernel_type = 'weiland'
trainable_kernel_delta = False
normalisation_inner = 'exp'
inducing_point_variational_parameterisation_type = 'gaussian'
init_min_seps = False
all_set_sizes = [2]
R_per_dim = 12
fix_inducing_point_locations = False
fix_non_swap = True
include_pi_1_tilde = False
include_pi_u_tilde = True
monte_carlo_gp_samples = 1024
remove_uniform = False
resume_path = None
shared_emission_distribution = False
shared_swap_function = False
swap_type = 'cue_dim_only'
symmetricality_constraint = True

min_sep = torch.pi / 7.1

num_models = 1


swap_model, D, delta_dimensions = setup_model_whole(
    num_models = num_models,
    swap_type = swap_type,
    kernel_type = kernel_type,
    emission_type = emission_type,
    all_set_sizes = all_set_sizes,
    remove_uniform = remove_uniform,
    include_pi_u_tilde = include_pi_u_tilde,
    trainable_kernel_delta = trainable_kernel_delta,
    R_per_dim = R_per_dim,
    fix_non_swap = fix_non_swap,
    include_pi_1_tilde = include_pi_1_tilde,
    fix_inducing_point_locations = fix_inducing_point_locations,
    symmetricality_constraint = symmetricality_constraint,
    shared_swap_function = shared_swap_function,
    shared_emission_distribution = shared_emission_distribution,
    all_min_seps = torch.tensor([[min_sep, min_sep]]),
    inducing_point_variational_parameterisation_type=inducing_point_variational_parameterisation_type,
    normalisation_inner= normalisation_inner,
    num_variational_samples = monte_carlo_gp_samples,
    num_importance_sampling_samples = monte_carlo_gp_samples,
    device=device,
    resume_path = None,
)


batch_size = 256
num_samples = 1
flip_augmentation = True

opt = torch.optim.Adam(swap_model.parameters(), lr = 0.01)

swap_model.cuda()

all_llh = []


infinite_data_mode = True


for i in tqdm(range(10_000)):

    if infinite_data_mode or i == 0:

        trial_information = task.generate_trial_information(batch_size=batch_size, num_samples=num_samples)

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


        cued_idx = trial_information.task_variable_information['cued_item_idx']

        all_probe_values = rectify_angles(trial_information.task_variable_information['probe_features'])
        probe_values = torch.zeros_like(all_probe_values)
        probe_values[torch.arange(batch_size),0] = all_probe_values[torch.arange(batch_size),cued_idx]
        probe_values[torch.arange(batch_size),1] = all_probe_values[torch.arange(batch_size),1-cued_idx]
        probe_deltas = rectify_angles(probe_values - probe_values[:,[0]])
        deltas_batch = probe_deltas.unsqueeze(-1).unsqueeze(0).repeat_interleave(num_models, 0)

        all_report_values = rectify_angles(trial_information.task_variable_information['report_features'])
        report_values = torch.zeros_like(all_report_values)
        report_values[torch.arange(batch_size),0] = all_report_values[torch.arange(batch_size),cued_idx]
        report_values[torch.arange(batch_size),1] = all_report_values[torch.arange(batch_size),1-cued_idx]

        responses_cart = novel_samples_dict['samples'].squeeze(1)
        responses = torch.arctan2(responses_cart[:,1], responses_cart[:,0]).unsqueeze(-1)

        errors = rectify_angles(responses - report_values)
        errors_batch = errors.unsqueeze(0).repeat_interleave(num_models, 0)


    if flip_augmentation:
        augmentation_map_deltas = (2 * (torch.rand_like(deltas_batch) < 0.5).int()) - 1
        deltas_batch = deltas_batch * augmentation_map_deltas.to(dtype = deltas_batch.dtype, device = deltas_batch.device)

        augmentation_map_errors = (2 * (torch.rand_like(errors_batch) < 0.5).int()) - 1
        errors_batch = errors_batch * augmentation_map_errors.to(dtype = errors_batch.dtype, device = deltas_batch.device)

    training_info = swap_model.get_elbo_terms(deltas = deltas_batch.cuda(), data = errors_batch.cuda(), max_variational_batch_size = 64)

    opt.zero_grad()

    selected_beta = 1.0
    total_elbo = (training_info['total_log_likelihood'] - selected_beta * training_info['kl_term'])
    total_loss = - total_elbo

    total_loss.sum().backward()
    opt.step()

    all_llh.append(training_info['total_log_likelihood'].item())

    torch.cuda.empty_cache()

    if i % 100 == 0:

        plt.close('all')

        fig, axes = plt.subplot_mosaic(
            '''
            AABB
            CCDD
            CCDD
            ''', figsize = (10, 10)
        )

        axes['A'].plot(all_llh)



        cued_idx = trial_information.task_variable_information['cued_item_idx']
        all_probe_values = rectify_angles(trial_information.task_variable_information['probe_features'])
        probe_values = torch.zeros_like(all_probe_values)
        probe_values[torch.arange(batch_size),0] = all_probe_values[torch.arange(batch_size),cued_idx]
        probe_values[torch.arange(batch_size),1] = all_probe_values[torch.arange(batch_size),1-cued_idx]
        probe_deltas = rectify_angles(probe_values - probe_values[:,[0]])
        deltas_batch = probe_deltas.unsqueeze(-1).unsqueeze(0).repeat_interleave(num_models, 0)
        all_report_values = rectify_angles(trial_information.task_variable_information['report_features'])
        report_values = torch.zeros_like(all_report_values)
        report_values[torch.arange(batch_size),0] = all_report_values[torch.arange(batch_size),cued_idx]
        report_values[torch.arange(batch_size),1] = all_report_values[torch.arange(batch_size),1-cued_idx]
        responses_cart = novel_samples_dict['samples'].squeeze(1)
        responses = torch.arctan2(responses_cart[:,1], responses_cart[:,0]).unsqueeze(-1)
        errors = rectify_angles(responses - report_values)
        errors_batch = errors.unsqueeze(0).repeat_interleave(num_models, 0)
        axes['B'].hist(errors[:,1], 50, alpha = 0.4)
        axes['B'].hist(errors[:,0], 50, alpha = 0.4)


        inference_info = swap_model.inference_on_grid(set_size=2, grid_count=100)
        one_dimensional_grid = inference_info['one_dimensional_grid']
        all_grid_points = inference_info['all_grid_points']
        mean_surface = inference_info['mean_surface']
        std_surface = inference_info['std_surface']
        function_samples_on_grid = inference_info['function_samples_on_grid']

        q = 0
        qth_mean_surface = mean_surface[q]
        qth_lower_surface, qth_upper_surface = qth_mean_surface - 2 * std_surface[q], qth_mean_surface + 2 * std_surface[q]

        display_pi_u_tilde = swap_model.generative_model.swap_function.pi_u_tilde_holder['2'].pi_tilde.detach().cpu()

        surface_color = 'blue'

        axes['C'].plot(one_dimensional_grid, qth_mean_surface, color = 'blue')[0].get_color()
        axes['C'].fill_between(one_dimensional_grid, qth_lower_surface, qth_upper_surface, color = surface_color, alpha = 0.2)
        axes['C'].plot([-torch.pi, torch.pi], [display_pi_u_tilde[q].item(), display_pi_u_tilde[q].item()], surface_color, linestyle= '-.', linewidth = 3)

        y_bot, y_top = axes['C'].get_ylim()
        sep = torch.pi/4.1
        axes['C'].plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
        axes['C'].plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
        axes['C'].set_ylim(y_bot, y_top)
        axes['C'].set_xlim(-torch.pi, torch.pi)

        if 'cyclical' in run_name:
            cued_probe_sq_distance = rectify_angles(deltas_batch).square()

            true_swap_func = -0.5 * (deltas_batch.square() / (2.0 + 2e-5))
            axes['C'].scatter(deltas_batch[0].flatten(), true_swap_func.flatten(), color = 'red', alpha = 0.4, s = 5)


        standard_swap_model_simplex_plots(training_info['priors'][0].detach().cpu().numpy(), axes['D'])

        fig.savefig(os.path.join(save_path, 'fit'))
        torch.save(swap_model.state_dict(), os.path.join(save_path, 'swap_model.mdl'))
