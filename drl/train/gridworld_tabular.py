import torch

import matplotlib.pyplot as plt

from drl.rl_envs.gridworld.grid_world import GridWorld
from drl.rl_envs.gridworld.cell_types import (
    RewardingTerminalCell,
)

from drl.rl_agents.gridworld.tabular_actor_critic import TabularActorModel, EpsGreedyActorModel, TabularGridWorldCriticModel


grid_text = """
#################
#.......#.......#
#E.D.G..#..S...M#
#..#............#
#################
"""
# grid_text = """
# #########
# #E..S..D#
# #########
# """
env = GridWorld(
    grid_text= grid_text,
    grid_config={
        'E': (RewardingTerminalCell, {'mean': +40.0, 'std': 0.5}),
        'G': (RewardingTerminalCell, {'mean': +4.0, 'std': 0.5}),
        'M': (RewardingTerminalCell, {'mean': +4.0, 'std': 0.5}),
        'D': (RewardingTerminalCell, {'mean': -40.0, 'std': 0.5}),
    }
)


batch_size = 64
remaining_terminations = 64 * 64 * 64
gamma = 0.95
lr = 1.0

fig_path = '/homes/pr450/repos/research_projects/sampling_ddpm/drl/train'

actor_model = TabularActorModel(num_states=len(env.cells))
#actor_model = EpsGreedyActorModel(num_states=len(env.cells))
#assert len(list(actor_model.parameters())) == 0

critic_model = TabularGridWorldCriticModel(num_states=len(env.cells), discount_factor=gamma)
optim = torch.optim.SGD(list(actor_model.parameters()) + list(critic_model.parameters()), lr = lr)


current_states = env.start_trials(batch_size)
all_step_rewards = []
all_critic_loss = []
all_actor_loss = []
ticker = 0


while remaining_terminations > 0:

    optim.zero_grad()

    ticker += 1

    # Take step in env
    if actor_model.requires_q_values:
        current_state_values = critic_model.get_q_values(current_states).detach()
        action_probs, action_choices = actor_model.choose_actions(current_states, current_state_values)       # pi(.|s) and a
    else:
        action_probs, action_choices = actor_model.choose_actions(current_states)       # pi(.|s) and a
    new_transitions = env.take_step(actions = action_choices, current_states = current_states)

    # Get critic loss, which relies on the next actions a', but with a stop-gradient
    with torch.no_grad():
        if actor_model.requires_q_values:
            next_state_values = critic_model.get_q_values(new_transitions.non_terminal_new_state_ids).detach()
            _, next_actions = actor_model.choose_actions(new_transitions.non_terminal_new_state_ids, next_state_values)    # a'
        else:
            _, next_actions = actor_model.choose_actions(new_transitions.non_terminal_new_state_ids)    # a'

    td1_info = critic_model.get_targets(transition_batch = new_transitions, next_actions = next_actions)
    critic_loss = critic_model.get_loss(td1_info = td1_info)
    print(critic_loss.item())

    # Update actor parameters
    actor_loss = actor_model.get_loss(
        action_probs = action_probs,
        action_choices = action_choices,
        current_state_action_values = td1_info.previous_q.detach()
    )

    # Run optimisation
    total_loss = critic_loss + actor_loss
    total_loss.backward()
    optim.step()
 
    all_step_rewards.append(new_transitions.transition_rewards[new_transitions.terminal].mean().item())
    all_critic_loss.append(critic_loss.item())
    all_actor_loss.append(actor_loss.item())

    if ticker % 100 == 0 or remaining_terminations < 0:
        fig, axes = plt.subplots(4, 2, figsize = (20, 10))
        axes[0,0].plot(all_step_rewards)
        axes[0,0].set_title('all_step_rewards')
        axes[1,0].plot(all_critic_loss)
        axes[1,0].set_title('all_critic_loss')
        axes[2,0].plot(all_actor_loss)
        axes[2,0].set_title('all_actor_loss')

        env.display_q_values(q_values = critic_model.values.detach().cpu().numpy(), axes = axes[:,1])
        fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/drl/train/gridworld_test.png')
        plt.close('all')


    # Update states being tracked, including potentially including more in
    remaining_terminations = remaining_terminations - new_transitions.num_terminal
    replenish_target_batch_size = min(batch_size, remaining_terminations)
    current_states = env.fill_in_trials(new_transitions, replenish_target_batch_size)
    print(remaining_terminations.item())

