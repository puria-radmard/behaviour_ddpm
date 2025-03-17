from analysis.decoder_analysis.shared_setup import *

if analysis_args.precue:
    fixed_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
    decoded_timestep = -5    # 5 steps before cue is presented...
else:
    raise NotImplementedError

random_choice = analysis_args.random_choice


decoder = ReportFeatureProjectors(dim_K = 5, dim_R = num_neurons)
decoder.cuda()

optim = torch.optim.Adam(decoder.parameters(), lr = 0.1)

save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')
    try:
        decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.mdl')))
    except (FileNotFoundError, RuntimeError):
        print('NO SAVED MODEL TO RESUME FROM')

analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))

ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps

decoding_variable = 'report_features_cart'
context_variable = 'probe_features_cart'

display_model_idx = 0

all_positional_decoder_losses = np.zeros([num_opt_steps])
all_shuffled_positional_decoder_losses = np.zeros([num_opt_steps])

num_samples = 1

infinite_data_mode = True


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

        relevant_activity = novel_samples_prep_dicts[1]['preparatory_trajectory'].squeeze(1)[:,decoded_timestep,:]  # [batch dim (128)]

    report = trial_information.task_variable_information[decoding_variable].cuda().to(relevant_activity.dtype)
    probe = trial_information.task_variable_information[context_variable].cuda().to(relevant_activity.dtype)
    output = decoder.get_mixture_model_means_precue(report_features=report, probe_features=probe)
    

    with torch.no_grad():
        shuffled_probe = probe[torch.randperm(probe.shape[0])]
        shuffled_output = decoder.get_mixture_model_means_precue(report_features=report, probe_features=shuffled_probe)

        all_trials_shuffled_loss = (relevant_activity - shuffled_output).square().sum(-1).sqrt()  # XXX: shape!
        all_shuffled_positional_decoder_losses[i] = all_trials_shuffled_loss.mean().item()

    all_trials_loss = (relevant_activity - output).square().sum(-1).sqrt()  # batch
    loss = all_trials_loss.mean()  # XXX: shape!

    optim.zero_grad()
    loss.backward()
    optim.step()

    all_positional_decoder_losses[i] = loss.item()

    if i % 50 == 0:

        plt.close('all')


        ### PLOT TRAINING LOSSES
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))

        zoomed_positional_decoder_losses_line = all_positional_decoder_losses[i+1-50:i+1]
        zoomed_shuffled_positional_decoder_losses_line = all_shuffled_positional_decoder_losses[i+1-50:i+1]
        positional_decoder_losses_line = all_positional_decoder_losses[:i+1]
        shuffled_positional_decoder_losses_line = all_shuffled_positional_decoder_losses[:i+1]

        axes[0,0].plot(zoomed_positional_decoder_losses_line)
        axes[0,1].plot(zoomed_shuffled_positional_decoder_losses_line)
        axes[1,0].plot(positional_decoder_losses_line)
        axes[1,1].plot(shuffled_positional_decoder_losses_line)
        
        axes[0,0].set_title('zoomed_positional_decoder_losses_line')
        axes[0,1].set_title('zoomed_shuffled_positional_decoder_losses_line')
        axes[1,0].set_title('positional_decoder_losses_line')
        axes[1,1].set_title('shuffled_positional_decoder_losses_line')

        fig.savefig(os.path.join(save_path, 'loss'))

        
        ### HIST SHUFFLED VS REAL LOSSES!
        fig, axes = plt.subplots(1, figsize = (8, 8))
        axes.hist(all_trials_loss.detach().cpu().numpy(), 64, alpha = 0.3, label = 'real euclidean distances')
        axes.hist(all_trials_shuffled_loss.detach().cpu().numpy(), 64, alpha = 0.3, label = 'shuffled probes euclidean distances')
        axes.legend()
        fig.savefig(os.path.join(save_path, 'loss_vs_shuffled'))

        
        torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.mdl'))

