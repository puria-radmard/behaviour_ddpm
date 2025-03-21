from analysis.decoder_analysis.shared_setup import *


first_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
diffusion_duration = ddpm_model.T       # Not used here though!


magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=first_delay_duration)
kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
kl_colors_scalarMap.set_array([])


mode_generator = CuedIndexDependentReportFeatureProjector(dim_K = 8, dim_R = num_neurons, precue_duration = first_delay_duration, postcue_duration = diffusion_duration)
mode_generator.cuda()

p_misbind = torch.nn.Parameter(torch.tensor([]))

all_parameters = list(mode_generator.parameters()) + [p_misbind]

optim = torch.optim.Adam(all_parameters, lr = 0.01)




save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')
    try:
        mode_generator.load_state_dict(torch.load(os.path.join(save_path, 'mode_generator.mdl')))
    except (FileNotFoundError, RuntimeError):
        print('NO SAVED MODEL TO RESUME FROM')
analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))




ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps


# Only accept trials where p_correct > 0.7 or < 0.3
p_decision_thres = 0.7


all_positional_decoder_losses = np.zeros([num_opt_steps, first_delay_duration])
all_reversed_positional_decoder_losses = np.zeros([num_opt_steps, first_delay_duration])

infinite_data_mode = True
num_samples = 1


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

        first_delay_activity = novel_samples_prep_dicts[1]['preparatory_trajectory'].squeeze(1)  # [batch dim, duration]
        
        import pdb; pdb.set_trace(header = 'get p_swap etc and filter trials here!')


    report = trial_information.task_variable_information['report_features_cart'][selected_trials_mask].cuda().to(first_delay_activity.dtype)
    cued_idx = trial_information.task_variable_information['cued_item_idx'][selected_trials_mask].cuda().to(first_delay_activity.dtype)          # XXX: no difference at all
    output = mode_generator.get_mixture_model_means_precue(report_features=report, probe_features=cued_idx)            # [B, T, R]
    all_trials_loss = (first_delay_activity - output).square().sum(-1).sqrt()  # [B, T]

    with torch.no_grad():
        
        reversed_report = report[:,[1,0],:]
        reversed_output = mode_generator.get_mixture_model_means_precue(report_features=reversed_report, probe_features=cued_idx)
        all_trials_reversed_loss = (first_delay_activity - reversed_output).square().sum(-1).sqrt()  # [B, T]

        all_positional_decoder_losses[i] = all_trials_loss.mean(0).tolist()
        all_reversed_positional_decoder_losses[i] = all_trials_reversed_loss.mean(0).tolist()

    loss = all_trials_loss.mean()
    optim.zero_grad()
    loss.backward()
    optim.step()


    if i % 50 == 0:

        plt.close('all')


        ### PLOT TRAINING LOSSES
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))
        
        
        for j in range(first_delay_duration):
            color = kl_colors_scalarMap.to_rgba(j + 1)

            zoomed_positional_decoder_losses_line = all_positional_decoder_losses[i+1-50:i+1,j]
            zoomed_reversed_positional_decoder_losses_line = all_reversed_positional_decoder_losses[i+1-50:i+1,j]
            positional_decoder_losses_line = all_positional_decoder_losses[:i+1,j]
            reversed_positional_decoder_losses_line = all_reversed_positional_decoder_losses[:i+1,j]

            axes[0,0].plot(zoomed_positional_decoder_losses_line, color = color)
            axes[0,1].plot(zoomed_reversed_positional_decoder_losses_line, color = color)
            axes[1,0].plot(positional_decoder_losses_line, color = color)
            axes[1,1].plot(reversed_positional_decoder_losses_line, color = color)
            
            axes[0,0].set_title('zoomed_positional_decoder_losses_line')
            axes[0,1].set_title('zoomed_reversed_positional_decoder_losses_line')
            axes[1,0].set_title('positional_decoder_losses_line')
            axes[1,1].set_title('reversed_positional_decoder_losses_line')

        fig.savefig(os.path.join(save_path, 'loss'))

        
        ### HIST reversed VS REAL LOSSES!
        fig, hist_axes = plt.subplots(first_delay_duration, 1, figsize = (8, 4 * first_delay_duration))
        for t, axes in enumerate(hist_axes):
            axes.hist(all_trials_loss[:,t].detach().cpu().numpy(), 64, alpha = 0.3, label = 'real euclidean distances')
            axes.hist(all_trials_reversed_loss[:,t].detach().cpu().numpy(), 64, alpha = 0.3, label = 'reversed probes euclidean distances')
            axes.set_title(t)
        axes.legend()
        fig.savefig(os.path.join(save_path, 'loss_vs_shuffled'))

        
        torch.save(mode_generator.state_dict(), os.path.join(save_path, 'mode_generator.mdl'))

