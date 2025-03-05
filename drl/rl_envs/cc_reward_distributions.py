from torch.distributions import Gamma
import torch


# target_rewards_distribution_probs_x_axis = torch.linspace(-5, +5, 50)


def GaussianMixture(weights, means, stds):
    reward_distribution_mix = torch.distributions.Categorical(torch.tensor(weights))
    reward_distribution_comp = torch.distributions.Normal(torch.tensor(means), torch.tensor(stds))
    reward_distribution = torch.distributions.MixtureSameFamily(reward_distribution_mix, reward_distribution_comp)
    return reward_distribution


