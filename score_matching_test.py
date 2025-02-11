import torch
from torch import Tensor as _T

import matplotlib.pyplot as plt

from tqdm import tqdm

from ddpm.tasks.distribution import DiagonalGaussianMixtureDistributionInformation


rad = 3.0
comp_weights = torch.tensor([1.0, 1.0, 1.0]) / 3
initial_means = rad * torch.tensor([[1.0, 1.0], [-1.0, -1.0], [-2.0, 2.0]])
# initial_means = initial_means - initial_means.mean(0, keepdim=True)
initial_std = 1.0
initial_std2 = torch.ones_like(initial_means) * initial_std ** 2
num_data = 500
selected_comps = torch.randint(0, len(initial_means), [num_data])
initial_data = initial_means[selected_comps].unsqueeze(0)  # [B = 1, S, 2]
initial_data = initial_data + initial_std * torch.randn_like(initial_data)

starting_sigma2 = 0.08
ultimate_sigma2 = 0.08
num_timesteps = 60
sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_timesteps)
std_schedule = torch.sqrt(sigma2x_schedule)
incremental_modulation_schedule = torch.sqrt(1.0 - sigma2x_schedule)

# One-shot noising
a_t_schedule = torch.cumprod(incremental_modulation_schedule, 0)  # t = 1, ..., T
root_b_t_schedule = torch.sqrt(1.0 - torch.square(a_t_schedule))  # t = 1, ..., T
b_t_schedule = root_b_t_schedule.square()

epsilon_actual = torch.randn(num_timesteps, *initial_data.shape)

trajectory = [initial_data]
for t, eps in enumerate(epsilon_actual):  # Forward time

    # Apply forward:
    #   dx = -0.5 sigma_t^2 x dt + sigma_t dw   ===>    f(x, t) = -0.5 sigma_t^2 x; g(t) = sigma_t
    trajectory.append(
        (trajectory[-1] * incremental_modulation_schedule[t])
        + (std_schedule[t] * eps)
    )


distn = DiagonalGaussianMixtureDistributionInformation(
    weights = comp_weights[None,None], 
    means = initial_means[None,None], 
    vars = initial_std2[None,None],
    sample_size = 2
)


# def evaluate_true_score(x: _T, mu: _T, s2: _T, pi: _T, bt: _T, at: _T):
#     """
#     B = batch size
#     S = num samples
#     M = num modes
#     D = dimensionality of data

#     x: reverse process shaped [B, S, D]
#     mu: mode means shaped [B, M, D]
#     s2: (shared) mode std in all directions, scalar
#     pi: weights sized [B, M]
#     bt, at2: scheule entries scalars

#     return: [B, S, D]
#     """
    
#     # useful stuff [B, S, M, D]
#     error_from_mode = (x.unsqueeze(-2) - (at * mu).unsqueeze(1))
#     score_linear_weighter = - 0.5 * error_from_mode / (bt + (at**2) * s2)

#     # scalar
#     dim = x.shape[-1]
#     gaussian_normalisation = 1 / torch.pow(2 * torch.pi * s2, torch.tensor(dim / 2))

#     # denominator = pdf [B, S]
#     gaussian_exponent = (score_linear_weighter * error_from_mode).sum(-1)   # [B, S, M]
#     scaled_pdf_by_mode = gaussian_exponent.exp() * pi.unsqueeze(1) * gaussian_normalisation # [B, S, M]
#     pdf = scaled_pdf_by_mode.sum(-1)    # [B, S]

#     # numerator = grad [B, S, M, D] -> [B, S, D]
#     pdf_grad = 2 * (score_linear_weighter * scaled_pdf_by_mode.unsqueeze(-1)).sum(-2)

#     score = pdf_grad / pdf.unsqueeze(-1)
    
#     if score.isnan().any() or score.isinf().any():
#         print(f'score has bad values! {t_idx}')
    
#     return score



rev_trajectory = [trajectory[-1].unsqueeze(-2)]
for t_idx in tqdm(range(1, num_timesteps+1)):

    # score = evaluate_true_score(
    score = distn.calculate_score(
        x_t = rev_trajectory[-1],
        a_t = a_t_schedule[[-t_idx]],
        b_t = b_t_schedule[[-t_idx]],
    )


    eps = torch.randn(*score.shape)

    # XXX: why did this not work?
    # next_rev_trajectory_diff = - sigma2x_schedule[-t_idx] * (
    #     0.5 * rev_trajectory[-1] 
    #     + score
    # )
    # next_rev_trajectory_mean = rev_trajectory[-1] + next_rev_trajectory_diff
    next_rev_trajectory_mean = (rev_trajectory[-1] + sigma2x_schedule[-t_idx] * score) / incremental_modulation_schedule[-t_idx]

    rev_trajectory.append(
        next_rev_trajectory_mean
        + (std_schedule[-t_idx] * eps)
    )


x_t_for = torch.stack(trajectory[1:], 2)[0]         # timestep, data, dim
x_t_rev = torch.concat(rev_trajectory[1:], -2)[0]

fig, axes = plt.subplots(2, 2, figsize = (20, 20))
axes = axes.flatten()

axes[0].scatter(*initial_data.numpy().T)
axes[0].scatter(*x_t_for[:,-1].numpy().T)
axes[0].plot(*x_t_for[0].numpy().T)
axes[0].set_xlim(-3 * rad, 3 * rad)
axes[0].set_ylim(-3 * rad, 3 * rad)


# axes[1].scatter(*trajectory[-1].numpy().T)
axes[1].scatter(*x_t_rev[:,-1].numpy().T)
axes[1].plot(*x_t_rev[0].numpy().T)
axes[1].set_xlim(-3 * rad, 3 * rad)
axes[1].set_ylim(-3 * rad, 3 * rad)



distn.display_final_score(-3 * rad, 3 * rad, 3 * rad, -3 * rad, axes[1], dict(color='g', alpha = 0.2))
distn.display_final_score(-3 * rad, 3 * rad, 3 * rad, -3 * rad, axes[2], dict(color='g'))
# x, y = torch.meshgrid(torch.linspace(-3 * rad, 3 * rad, 20), torch.linspace(-3 * rad, 3 * rad, 20)) 
# score = evaluate_true_score(
#     x = torch.stack([x, y], -1), mu = initial_means.unsqueeze(0), s2 = initial_std ** 2,
#     pi = comp_weights.unsqueeze(0), bt = b_t_schedule[0], at = a_t_schedule[0]
# )
# u, v = score.permute(2, 0, 1)
# axes[2].quiver(x, y, u, v, color='g') 
# axes[1].quiver(x, y, u, v, color='g', alpha = 0.2) 


    
axes[3].plot(range(1, 1+num_timesteps), sigma2x_schedule, label = 'sigma2x_schedule')
axes[3].plot(range(1, 1+num_timesteps), std_schedule, label = 'std_schedule')
axes[3].plot(range(1, 1+num_timesteps), incremental_modulation_schedule, label = 'incremental_modulation_schedule')
axes[3].plot(range(1, 1+num_timesteps), a_t_schedule, label = 'a_t_schedule')
axes[3].plot(range(1, 1+num_timesteps), b_t_schedule, label = 'b_t_schedule')
axes[3].legend()

fig.savefig('asdf')
