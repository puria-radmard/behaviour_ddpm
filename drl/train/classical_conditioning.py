from drl.setup import *

from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import kstest
from purias_utils.util.logging import configure_logging_paths


from drl.util import plot_model_schedules, make_classical_conditioning_stimuli, make_reward_distributions

import os



(
    all_reward_distributions,
    all_target_rewards_distribution_probs_x_axes,
    all_target_rewards_distribution_probs,
    all_numpy_friendly_reward_distributions,
) = make_reward_distributions(all_reward_distribution_configs)


all_consistency_ks_stats = np.zeros([num_batches, wait_time + time_between_cs_and_us + time_after_us - 1, len(all_reward_distributions)])
all_reward_ks_stats = np.zeros([num_batches, wait_time + time_between_cs_and_us + time_after_us - 1, len(all_reward_distributions)])


(
    stimulus_array,
    num_task_timesteps,
    s_t,
    s_t_plus_1,
    s_all,
    input_t,
    input_t_plus_1,
    input_all,
) = make_classical_conditioning_stimuli(
    wait_time,
    time_between_cs_and_us,
    time_after_us,
    batch_size,
    num_diffusion_timesteps,
    device,
)


steps_to_reward = time_between_cs_and_us - stimulus_array - 1
q_value_scaler = (gamma ** steps_to_reward).cpu().numpy()
q_value_scaler[stimulus_array.cpu().numpy() == -1] = 1.0

prop_counter_balance = gamma ** time_between_cs_and_us



_, save_base, _ = configure_logging_paths(
    save_base, log_suffixes=[], index_new=True
)
plot_model_schedules(ddpm_model, os.path.join(save_base, "sigma_schedule_unrolling.png"))
args.write_to_yaml(os.path.join(save_base, "args.yaml"))



for batch in tqdm(range(num_batches)):

    if batch < start_freezing_batch:
        target_sampler = ddpm_model
    else:
        if batch % freezing_frequency == 0:
            target_sampler = make_model(len(all_reward_distributions))
            target_sampler.load_state_dict(copy.deepcopy(ddpm_model.state_dict()))
            for parameter in target_sampler.parameters():
                parameter.requires_grad = False
            

    optim.zero_grad()

    # Reward given when stimulus_array == time_between_cs_and_us
    reward = torch.zeros(*s_t.shape, len(all_reward_distributions)).float()
    for i_r, reward_distribution in enumerate(all_reward_distributions):
        rewards = reward_distribution.sample([batch_size])              # [batch_size]
        reward[:,wait_time + time_between_cs_and_us - 1, i_r] = rewards
    reward = reward.to(device)
        

    for _ in range(opt_steps_per_batch):

        # Generate diffusion targets
        # z_t_plus_1_samples_dict['samples'] of shape [batch, time-1, 1]
        with torch.no_grad():
            z_t_plus_1_samples_dict = target_sampler.generate_samples(
                network_input = input_t_plus_1,
                samples_shape = [batch_size, num_task_timesteps - 1],
            )

            # Perform Bellman backup
            unrolled_z_samples = reward + gamma * z_t_plus_1_samples_dict['samples'].to(device)

        # Do diffusion training step on the Bellman backup
        forward_process = ddpm_model.noise(
            x_0=unrolled_z_samples
        )
        epsilon_hat_dict = ddpm_model.residual(
            x_samples = forward_process["x_t"],
            network_input = input_t,
        )

        # Optimisation step
        stepwise_losses = (epsilon_hat_dict['epsilon_hat'] - forward_process["epsilon"]).square().mean(-1).mean(0).mean(0)
        total_loss = stepwise_losses.mean()
        
        total_loss.backward()
        optim.step()

        all_stepwise_losses[batch,:] += stepwise_losses.detach().cpu().numpy() / opt_steps_per_batch
        all_mean_losses[batch] += total_loss.item() / opt_steps_per_batch



    with torch.no_grad():
        # novel_samples['samples'] of shape [batch, time, 1]
        novel_samples = ddpm_model.generate_samples(
            network_input = input_all,
            samples_shape = [batch_size, num_task_timesteps],
        )
    
    for i_r in range(len(all_reward_distributions)):
        for i in range(wait_time + time_between_cs_and_us + time_after_us - 1):
            all_consistency_ks_stats[batch,i,i_r] = kstest(
                novel_samples['samples'][:,i,i_r].cpu().numpy(),
                (reward[:,i,0] + gamma * novel_samples['samples'][:,i+1,i_r]).cpu().numpy()
            ).statistic

            discount_amount_i = q_value_scaler[i]
            try:
                all_reward_ks_stats[batch,i,i_r] = kstest(
                    novel_samples['samples'][:,i,i_r].cpu().numpy() / discount_amount_i,
                    all_numpy_friendly_reward_distributions[i_r]
                ).statistic
            except ValueError as e:
                pass

    if batch % logging_freq == 0:

        num_rows = 1 + len(all_reward_distributions)
        fig, axes = plt.subplots(num_rows, 3, figsize = (30, 10 * num_rows))

        # PLOT LOSSES
        axes[0,0].set_title("Individual timesteps MSE")
        axes[0,1].set_title("MSE averaged over timesteps")
        axes[0,1].plot(all_mean_losses[: batch + 1])
        for h, trace in enumerate(
            all_stepwise_losses[: batch + 1].T
        ):
            color = diffusion_timesteps_colors_scalarMap.to_rgba(h + 1)
            axes[0,0].plot(trace, color=color)
        cax = inset_axes(axes[0,0], width="30%", height=1.0, loc=3)
        plt.colorbar(
            diffusion_timesteps_colors_scalarMap,
            cax=cax,
            ticks=range(1, num_diffusion_timesteps, 10),
            orientation="vertical",
        )

        # FOR EACH DISTRIBUTION, DO 1D HISTOGRAMS AGAINST REWARD DIST, and plot KS stuff
        for i_row in range(len(all_reward_distributions)):
            axes[1 + i_row,2].plot(
                all_target_rewards_distribution_probs_x_axes[i_row].cpu().numpy(),
                all_target_rewards_distribution_probs[i_row].cpu().numpy(),
                label = 'true reward distribution',
                color = 'black'
            )

            # Plot value distribution
            for t in range(-wait_time, time_between_cs_and_us + time_after_us):   # time_since_stim
                adjusted_t = wait_time + t
                discount_amount = q_value_scaler[adjusted_t]
                
                relevant_preds = novel_samples['samples'][:,adjusted_t,i_row] / discount_amount    # recycling this
                
                t_color = task_timesteps_colors_scalarMap.to_rgba(t)
                t_label = f"{t} (div {round(discount_amount, 3)})"
                
                axes[1 + i_row,2].hist(relevant_preds.cpu().numpy(), 64, alpha = 0.7, histtype=u'step', density=True, label = t_label, color = t_color)

            if i_row == 0:
                axes[1 + i_row,2].legend(title = f'timesteps since CS+\n(adjusted for $\gamma =${gamma}), T = {time_after_us}') 
            axes[1 + i_row,2].set_title('Histogram of scaled z samples, y axis cropped')
            axes[1 + i_row,2].set_ylim(0, all_target_rewards_distribution_probs[i_row].max() * 3)

            # Plot temporal consistency
            for i in range(wait_time + time_between_cs_and_us + time_after_us - 1):
                i_color = task_timesteps_colors_scalarMap.to_rgba(i - wait_time)
                axes[1 + i_row,0].plot(all_consistency_ks_stats[: batch + 1,i,i_row], label = f'{i - wait_time}', color = i_color)
                axes[1 + i_row,1].plot(all_reward_ks_stats[: batch + 1,i,i_row], label = f'{i - wait_time}', color = i_color)

            if i_row == 0:
                cax = inset_axes(axes[1 + i_row,0], width="30%", height=1.0, loc=3)
                cax.set_title(f'timesteps since CS+, T = {time_between_cs_and_us}')
                plt.colorbar(
                    task_timesteps_colors_scalarMap,
                    cax=cax,
                    ticks=range(1, num_task_timesteps, 10),
                    orientation="vertical",
                )

            axes[1 + i_row,0].set_title('Sample-sample KS test for v.d. consistency across time')
            axes[1 + i_row,1].set_title('Sample-CDF KS test for v.d. consistency againsted true reward')


        # IF WE HAVE TWO DIMENSIONAL REWARDS, PLOT HISTOGRAM FOR THE JOINT SAMPLES
        if len(all_reward_distributions) == 2:

            for t in range(-wait_time, time_between_cs_and_us + time_after_us):   # time_since_stim
                adjusted_t = wait_time + t
                discount_amount = q_value_scaler[adjusted_t]
                
                relevant_preds = novel_samples['samples'][:,adjusted_t,:] / discount_amount    # recycling this
                
                t_color = task_timesteps_colors_scalarMap.to_rgba(t)
                
                counts,ybins,xbins,image = axes[0,2].hist2d(relevant_preds.cpu().numpy()[...,0], relevant_preds.cpu().numpy()[...,1], 64, alpha = 0.0)
                axes[0,2].contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3, alpha = 0.7, colors = [t_color], levels = 5)
        
    
        plt.savefig(os.path.join(save_base, "latest_log.png"))
        plt.close("all")

        torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.mdl"))
