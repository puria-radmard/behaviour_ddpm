from analysis.decoder_analysis.shared_setup import *
from torch import Tensor as _T


recall_epoch_duration = ddpm_model.sigma2xt_schedule.shape[0]
fixed_delay_duration = 5   # XXX remove hard coding!

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

decoder = CrossTemporalLinearDecoder(recall_epoch_duration, num_neurons)

optim = torch.optim.Adam(decoder.parameters())


all_positional_decoder_losses = np.zeros([num_opt_steps, recall_epoch_duration])
all_other_positional_decoder_losses = np.zeros([num_opt_steps, recall_epoch_duration])

num_samples = 1

infinite_data_mode = True

magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=recall_epoch_duration)
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

        diffusion_time_activity = novel_samples_dict['embedded_sample_trajectory'].squeeze(1).cuda()  # [batch, time (30), dim (128)]

    cued_idxes: _T = trial_information.task_variable_information['cued_item_idx'].int()
    if decode_cued:
        decoding_positions = cued_idxes
    else:
        decoding_positions = 1 - cued_idxes

    all_items = torch.arange(ctd_batch_size)

    # raise Exception('Do exclusion here!')
    if limit_to_further:
        all_items = all_items[((novel_samples_dict['samples'] - trial_information.task_variable_information['report_features_cart']).square().sum(-1).argmax(-1) == decoding_positions)]
        decoding_positions = decoding_positions[all_items]
        diffusion_time_activity = diffusion_time_activity[all_items]
        print(len(all_items) / ctd_batch_size)

    target = trial_information.task_variable_information[decoding_variable][all_items,decoding_positions].cuda().unsqueeze(1)
    output = decoder.decode_sequence(diffusion_time_activity)
    
    loss = (target.unsqueeze(0) - output).square().sum(-1).sqrt().mean(1)   # [model, timesteps]

    optim.zero_grad()
    loss.mean(0).sum().backward()
    optim.step()

    all_positional_decoder_losses[i] = loss.mean(0).tolist()

    with torch.no_grad():
        other_target = trial_information.task_variable_information[decoding_variable][all_items,1-decoding_positions].cuda().unsqueeze(1)
        other_losses = (other_target.unsqueeze(0) - output).square().sum(-1).sqrt().mean(1)
        all_other_positional_decoder_losses[i] = other_losses.mean(0).tolist()

    if i % 50 == 0:

        plt.close('all')


        ### PLOT TRAINING LOSSES
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))

        for j in range(recall_epoch_duration):
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

        
        
        ### PLOT DIFFERENCES IN LOSSES
        fig,axes = plt.subplots(1, figsize = (10, 6))
        for j in range(recall_epoch_duration):
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

        
        torch.save(ddpm_model.state_dict(), os.path.join(save_path, 'decoder.mdl'))

        np.save(os.path.join(save_path, 'recent_losses.mdl'), {'recent_losses': loss.detach().cpu().numpy(), 'other_losses': other_losses.detach().cpu().numpy()})
