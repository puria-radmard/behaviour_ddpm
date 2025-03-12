import os
import torch

import numpy as np

import matplotlib.cm as cmx
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from drl.rl_envs.gridworld.grid_world import GridWorld
from drl.rl_envs.gridworld.cell_types import (
    RewardingTerminalCell, BimodalRewardingTerminalCell, SpikyWallCell, WindyValveCell
)

from drl.rl_agents.gridworld.tabular_actor_critic import (
    TabularActorModel, EpsGreedyActorModel, TabularGridWorldCriticModel, FactoredEmbeddingDDPMCriticModel, UniqueEmbeddingDDPMCriticModel
)

from drl.util import plot_model_schedules

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


torch.set_default_device('cuda')


# grid_text = """
# #################
# #.......#.......#
# #E.D.G..#..S...G#
# #..#............#
# #################
# """
grid_text = """
XXXXX
X...X
X.X.X
XCX.X
X.6.X
XXSXX
XX.XX
XXXXX
"""
# grid_text = """
# E..S..D
# """
env = GridWorld(
    grid_text= grid_text,
    grid_config={
        'E': (BimodalRewardingTerminalCell, {'mean1': +10.0, 'std1': 2.0, 'mean2': -1.0, 'std2': 2.0, 'pi1': 0.4, 'pi2': 0.6}),
        'G': (RewardingTerminalCell, {'mean': +4.0, 'std': 2.0}),
        'A': (RewardingTerminalCell, {'mean': +4.0, 'std': 2.0}),
        'X': (SpikyWallCell, {}),
        '6': (WindyValveCell, {'valve_side': 'S', 'spit_sides': 'WE'}),
        'D': (BimodalRewardingTerminalCell, {'mean1': -5.0, 'std1': 2.0, 'mean2': 7.0, 'std2': 2.0, 'pi1': 0.4, 'pi2': 0.6}),
    }
)


batch_size = 64
remaining_terminations = 64 * 64 * 64
gamma = 0.95
lr = 0.01

fig_path = '/homes/pr450/repos/research_projects/sampling_ddpm/drl/train'

actor_model = TabularActorModel(num_states=len(env.cells))

# actor_model = EpsGreedyActorModel(epsilon = 1.0, num_states=len(env.cells))
# assert len(list(actor_model.parameters())) == 0

# critic_model = TabularGridWorldCriticModel(num_states=len(env.cells), discount_factor=gamma)

diffusion_time_embedding_size = 16
state_action_embedding_dim = 16

critic_model = UniqueEmbeddingDDPMCriticModel(
    reward_dim = 1,
    num_states = len(env.cells),
    discount_factor = gamma,
    num_actions_per_state = 4,
    state_action_embedding_dim = state_action_embedding_dim,
    diffusion_time_embedding_size = diffusion_time_embedding_size,
    device = 'cuda'
)

plot_model_schedules(critic_model.ddpm, os.path.join(fig_path, "sigma_schedule_unrolling.png"))


optim = torch.optim.Adam(list(actor_model.parameters()) + list(critic_model.parameters()), lr = lr)



current_states = env.start_trials(batch_size)
all_step_rewards = []
all_step_rewards_x = []
all_critic_loss = []
all_actor_loss = []

all_diffusion_stepwise_mse = np.zeros([0, num_diffusion_timesteps])

ticker = -1


while remaining_terminations > 0:

    ticker += 1

    # Take step in env
    if actor_model.requires_q_values:
        with torch.no_grad():
            current_state_values = critic_model.get_q_values(current_states).detach()
        action_probs, action_choices = actor_model.choose_actions(current_states, current_state_values)       # pi(.|s) and a
    else:
        try:
            action_probs, action_choices = actor_model.choose_actions(current_states)       # pi(.|s) and a
        except:
            import pdb; pdb.set_trace()
    new_transitions = env.take_step(actions = action_choices, current_states = current_states)

    # Get critic loss, which relies on the next actions a', but with a stop-gradient
    with torch.no_grad():
        if actor_model.requires_q_values:
            next_state_values = critic_model.get_q_values(new_transitions.non_terminal_new_state_ids).detach()
            _, next_actions = actor_model.choose_actions(new_transitions.non_terminal_new_state_ids, next_state_values)    # a'
        else:
            _, next_actions = actor_model.choose_actions(new_transitions.non_terminal_new_state_ids)    # a'

    td1_info = critic_model.get_targets(transition_batch = new_transitions, next_actions = next_actions)
    critic_loss, stepwise_losses = critic_model.get_loss(td1_info = td1_info, transition_batch = new_transitions)

    # Update actor parameters
    actor_loss = actor_model.get_loss(
        action_probs = action_probs,
        action_choices = action_choices,
        current_state_action_values = td1_info.previous_q.detach()
    )

    # Run optimisation
    total_loss = critic_loss + actor_loss
    
    optim.zero_grad()
    total_loss.backward()
    optim.step()
 
    if new_transitions.terminal.any():
        all_step_rewards.append(new_transitions.transition_rewards[new_transitions.terminal].mean().item())
        all_step_rewards_x.append(ticker)
    all_critic_loss.append(critic_loss.item())
    all_actor_loss.append(actor_loss.item())
    if stepwise_losses is not None:
        all_diffusion_stepwise_mse = np.concatenate([all_diffusion_stepwise_mse, stepwise_losses.detach().cpu()[None].numpy()], axis=0)


    # Update states being tracked, including potentially including more in
    remaining_terminations = remaining_terminations - new_transitions.num_terminal
    replenish_target_batch_size = min(batch_size, remaining_terminations)
    current_states = env.fill_in_trials(new_transitions, replenish_target_batch_size)
    print(ticker, remaining_terminations.item())

    if ticker % 1000 == 0 or remaining_terminations <= 0:

        if critic_model.is_distributional:
            
            ax_unit = 4
            ax_size = 0.75
            loss_margin_rows = 2
            num_rows = ax_unit * (env.height + loss_margin_rows)
            num_cols = ax_unit * env.width
            fig = plt.figure(constrained_layout=True, figsize = (num_cols * ax_size, num_rows * ax_size))
            gs = fig.add_gridspec(num_rows, num_cols)    # 3 added effective rows for all loss plots
            
            all_step_rewards_axes = fig.add_subplot(gs[:ax_unit*loss_margin_rows,:env.width])
            all_critic_loss_axes = fig.add_subplot(gs[:ax_unit*loss_margin_rows,env.width:2 * env.width])
            all_actor_loss_axes = fig.add_subplot(gs[:ax_unit*loss_margin_rows,2 * env.width:3 * env.width])
            
            all_stepwise_loss_axes = fig.add_subplot(gs[:ax_unit*loss_margin_rows,3 * env.width:])
            all_stepwise_loss_axes.set_title("Individual diffusion timesteps MSE")
            for h, trace in enumerate(
                all_diffusion_stepwise_mse.T
            ):
                color = diffusion_timesteps_colors_scalarMap.to_rgba(h + 1)
                all_stepwise_loss_axes.plot(trace, color=color)
            cax = inset_axes(all_stepwise_loss_axes, width="30%", height=1.0, loc=3)
            plt.colorbar(
                diffusion_timesteps_colors_scalarMap,
                cax=cax,
                ticks=range(1, num_diffusion_timesteps, 10),
                orientation="vertical",
            )

            all_histogram_axes = []
            for i_row in range(env.height):
                histogram_axes_row = []
                for i_col in range(env.width):
                    new_hist_ax = fig.add_subplot(
                        gs[
                            loss_margin_rows * ax_unit + i_row * ax_unit : loss_margin_rows * ax_unit + i_row * ax_unit + ax_unit, 
                            i_col * ax_unit : i_col * ax_unit + ax_unit
                        ],
                        # sharey = histogram_axes_row[-1] if i_col else None,   # Histogram gets tiny
                        sharex = all_histogram_axes[-1][i_col] if i_row else None
                    )
                    histogram_axes_row.append(new_hist_ax)
                all_histogram_axes.append(histogram_axes_row)
            all_histogram_axes = np.array(all_histogram_axes)

            with torch.no_grad():
                histable_values = critic_model.get_q_values(
                    states = torch.arange(len(env.cells)),
                    average = False
                )

            env.display_q_value_samples(q_values_samples = histable_values.detach(), axes = all_histogram_axes)
            
        else:
            fig, axes = plt.subplots(4, 2, figsize = (20, 10))
            all_step_rewards_axes = axes[0,0]
            all_critic_loss_axes = axes[1,0]
            all_actor_loss_axes = axes[2,0]

            env.display_q_values(q_values = critic_model.values.detach().cpu().numpy(), axes = axes[:,1])

        all_step_rewards_axes.plot(all_step_rewards_x, all_step_rewards)
        all_step_rewards_axes.set_title('all_step_rewards')
        all_critic_loss_axes.plot(all_critic_loss)
        all_critic_loss_axes.set_title('all_critic_loss')
        all_actor_loss_axes.plot(all_actor_loss)
        all_actor_loss_axes.set_title('all_actor_loss')

        fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/drl/train/gridworld_test.png')
        plt.close('all')
