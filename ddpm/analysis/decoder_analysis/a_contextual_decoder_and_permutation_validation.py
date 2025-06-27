from analysis.decoder_analysis.shared_setup import *

if analysis_args.diffusion_period:
    relevant_duration = ddpm_model.sigma2xt_schedule.shape[0]
    fixed_delay_duration = 20   # XXX remove hard coding!
else:
    fixed_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
    relevant_duration = fixed_delay_duration

random_choice = analysis_args.random_choice

decoder_type = analysis_args.decoder_type

if decoder_type == 'linear':
    decoder = CrossTemporalLinearDecoder(relevant_duration, num_neurons)
elif decoder_type == 'nonlinear':
    decoder = CrossTemporalNonLinearDecoder(relevant_duration, num_neurons)
elif decoder_type == 'contextual':
    decoder = CrossTemporalContextGatedDecoder(relevant_duration, num_neurons)
elif decoder_type == 'nonlinear_contextual':
    decoder = CrossTemporalNonLinearContextGatedDecoder(relevant_duration, num_neurons)
optim = torch.optim.Adam(decoder.parameters())


save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')
    decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.mdl')))

analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))

ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps

decoding_variable = analysis_args.decoding_variable
context_variable = analysis_args.context_variable

display_model_idx = 0


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

        if analysis_args.diffusion_period:
            relevant_activity = novel_samples_dict['embedded_sample_trajectory'].squeeze(1).cuda()  # [batch, time (30), dim (128)]
        else:
            relevant_activity = novel_samples_prep_dicts[1]['preparatory_trajectory'].squeeze(1)  # [batch, time (30), dim (128)]

    if random_choice:
        decoding_positions = torch.randint(0, 2, [ctd_batch_size]).int()
    else:
        decoding_positions = torch.zeros(ctd_batch_size).int()
    all_items = torch.arange(ctd_batch_size)

    target = trial_information.task_variable_information[decoding_variable][all_items,decoding_positions].cuda().unsqueeze(1)
    context = trial_information.task_variable_information[context_variable][all_items,decoding_positions].cuda().to(relevant_activity.dtype)
    output = decoder.decode_sequence(relevant_activity, context=context)
    
    all_trials_loss = (target.unsqueeze(0) - output).square().sum(-1).sqrt()[display_model_idx].detach().cpu().numpy()   # [batch, timestep]
    loss = (target.unsqueeze(0) - output).square().sum(-1).sqrt().mean(1)   # [model, timesteps]

    optim.zero_grad()
    loss.mean(0).sum().backward()
    optim.step()

    all_positional_decoder_losses[i] = loss.mean(0).tolist()

    with torch.no_grad():
        other_target = trial_information.task_variable_information[decoding_variable][all_items,1-decoding_positions].cuda().unsqueeze(1)
        other_all_trials_loss = (other_target.unsqueeze(0) - output).square().sum(-1).sqrt()[display_model_idx].detach().cpu().numpy()   # [batch, timestep]
        other_losses = (other_target.unsqueeze(0) - output).square().sum(-1).sqrt().mean(1)
        all_other_positional_decoder_losses[i] = other_losses.mean(0).tolist()

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

        

        ### PLOT CROSS TEMPORAL LOSSES
        fig, axes = plt.subplots(1, figsize = (8, 8))
        with torch.no_grad():
            cross_output, cross_losses = decoder.decode_cross_temporally(
                sequence = relevant_activity,
                targets = target.squeeze(1),
                context = context
            )

            
            ims = plt.imshow(cross_losses[0].mean(0).cpu().numpy(), origin='lower')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(ims, cax=cax, orientation='vertical')
        
        fig.savefig(os.path.join(save_path, 'ctd_loss'))

        
        ### PLOT DIFFERENCES IN LOSSES - over models
        fig,axes = plt.subplots(1, figsize = (10, 6))
        for j in range(relevant_duration):
            color = kl_colors_scalarMap.to_rgba(j + 1)

            loss_diffs = (other_losses[:,j] - loss[:,j]).detach().cpu().numpy()
            
            loss_x = (torch.ones(*loss_diffs.shape) * j)
            loss_x = (loss_x + (torch.randn_like(loss_x) * 0.2)).detach().cpu().numpy()

            axes.scatter(loss_x, loss_diffs, color = color)

            ttest = ttest_rel(loss_diffs, np.zeros_like(loss_diffs))
            if ttest.statistic < 0 or ttest.pvalue > 0.1:
                axes.text(s = f"{ttest.statistic}\n{ttest.pvalue}", x=j, y=loss_diffs.max())

        axes.plot(axes.get_xlim(), [0, 0], color = 'black')
        fig.savefig(os.path.join(save_path, 'swapped_loss'))


        ### PLOT DIFFERENCES IN LOSSES - over trials
        fig,axes = plt.subplots(1, figsize = (10, 6))

        for j in range(relevant_duration):
            color = kl_colors_scalarMap.to_rgba(j + 1)

            loss_diffs = (other_all_trials_loss[:,j] - all_trials_loss[:,j])
            loss_diff_mean = loss_diffs.mean()
            loss_diff_std = loss_diffs.std()

            loss_x = (torch.ones(*loss_diffs.shape) * j)
            loss_x = (loss_x + (torch.randn_like(loss_x) * 0.2)).detach().cpu().numpy()

            axes.scatter(loss_x, loss_diffs, alpha = 0.1, s = 2, color = color)
            axes.scatter([j], [loss_diff_mean], color = color)
            axes.plot([j, j], [loss_diff_mean - 2 * loss_diff_std, loss_diff_mean + 2 * loss_diff_std], color = color)

            ttest = ttest_rel(loss_diffs, np.zeros_like(loss_diffs))
            if ttest.statistic < 0 or ttest.pvalue > 0.1:
                axes.text(s = f"{ttest.statistic}\n{ttest.pvalue}", x=j, y=loss_diff_mean + 2 * loss_diff_std)

        axes.plot(axes.get_xlim(), [0, 0], color = 'black')
        fig.savefig(os.path.join(save_path, 'swapped_loss_across_trials'))


        
        torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.mdl'))

        np.save(os.path.join(save_path, 'recent_losses.mdl'), {'recent_losses': loss.detach().cpu().numpy(), 'other_losses': other_losses.detach().cpu().numpy()})
