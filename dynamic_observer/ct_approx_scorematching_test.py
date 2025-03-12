from __future__ import annotations

import torch

from dynamic_observer.model import *

from tqdm import tqdm

from numpy import mean


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    noise_schedule = LinearIncreaseNoiseSchedule(1.9, 1.9, duration = 5)
    # noise_schedule = CosineNoiseSchedule(mag = 0.9, duration = 30)

    score_func = FCScoreApproximator(
        sample_size = 2,
        hidden_layers = [16],
        input_tensor_size = 6,   # Flattened mu and Sigma
        input_repr_size = 16,
        input_hidden_layers = [16, 16],
        time_embedding_dim = 16,
        time_embedding_hidden_layers = None,
    )

    discretiser = EulerDiscretiser()
    
    diffmodel = ContinuousTimeScoreMatchingDiffusionModel(
        sample_dim=2, noise_schedule=noise_schedule, 
        score_approximator=score_func, discretiser=discretiser
    )

    batch_size = 64
    noising_batch_size = 32
    num_reverse_dynamics_steps = 100

    target_m0 = torch.tensor([-50.0, -10.0])[None,None].repeat(num_reverse_dynamics_steps-1, batch_size, 1)
    target_S0 = torch.tensor([[10.0, -3.0], [-3.0, 4.0]])[None,None].repeat(num_reverse_dynamics_steps-1, batch_size, 1, 1)
    target_S0_reshaped = target_S0.reshape(*target_S0.shape[:-2], -1)
    vector_stimulus = (torch.concat([target_m0, target_S0_reshaped], dim = -1), )

    target_S0_chol = torch.linalg.cholesky(target_S0[0,0])

    observations = 30.0 + torch.zeros(num_reverse_dynamics_steps-1, batch_size, 1)  # y
    projection_matrix = torch.tensor([[0.1961, -0.9806]])[None,None].repeat(num_reverse_dynamics_steps-1, batch_size, 1, 1)  # A
    observation_noise_covar = torch.tensor([[1.0, 0.0], [0.0, 1.0]])[None,None].repeat(num_reverse_dynamics_steps-1, batch_size, 1, 1)

    # Timepoints dont align, but this is the same stimulus throughout
    input_stimulus = (vector_stimulus[0][0].unsqueeze(0).repeat_interleave(noising_batch_size, 0),)

    all_losses = []
    all_losses_smoothed = []
    
    parameters = list(diffmodel.parameters())
    optimised = len(parameters) > 0
    if optimised:
        optim = torch.optim.Adam(diffmodel.parameters(), lr = 0.001)

    
    for opt_iter in tqdm(range(10000)):

        real_samples = torch.randn(batch_size, 2) @ target_S0_chol.T + target_m0[0,0]

        noised_timepoints, noised_information = noise_schedule.random_noise(
            real_samples, num_timepoints=noising_batch_size
        )

        approx_score = diffmodel.score_approximator.approximate_score(
            x_t = noised_information['x_t'],
            stimuli = input_stimulus,
            t = noised_timepoints
        )


        real_score = noised_information['conditional_score']
        loss = (real_score - approx_score).square().mean()

        if optimised:
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        all_losses.append(loss.item())
        all_losses_smoothed.append(mean(all_losses[-10:]))

        print(loss.item())

        if opt_iter % 100 == 0:

            base_samples = torch.randn(batch_size, 2)
            all_reverse_trajectories = diffmodel.run_unconditioned_reverse_dynamics(base_samples, vector_stimulus, num_reverse_dynamics_steps).detach().cpu().numpy()
            example_reverse_trajectories = all_reverse_trajectories[:5]
            end_reverse_samples = all_reverse_trajectories[:,-1,:]

            base_samples = torch.randn(batch_size, 2)
            all_conditioned_reverse_trajectories = diffmodel.run_conditioned_reverse_dynamics(
                base_samples, vector_stimulus, num_reverse_dynamics_steps, observations, projection_matrix, observation_noise_covar
            ).detach().cpu().numpy()
            example_conditioned_reverse_trajectories = all_conditioned_reverse_trajectories[:5]
            end_conditioned_reverse_samples = all_conditioned_reverse_trajectories[:,-1,:]
            
            real_samples = torch.randn(batch_size, 2) @ target_S0_chol.T + target_m0[0,0]
            all_forward_trajectories = diffmodel.run_forward_dynamics(real_samples, 100).detach().cpu().numpy()
            example_forward_trajectories = all_forward_trajectories[:5]
            end_forward_samples = all_forward_trajectories[:,-1,:]

            fig, axes = plt.subplots(1, 5, figsize = (25, 5))
            scaling_factor_time, scaling_factor = noise_schedule.summarise_noising_factor(100)
            axes[0].plot(scaling_factor_time.detach().cpu().numpy(), scaling_factor.detach().cpu().numpy())
            axes[0].set_ylim(0)

            axes[1].scatter(*end_reverse_samples.T)
            for example_reverse_trajectory in example_reverse_trajectories:
                axes[1].plot(*example_reverse_trajectory.T, alpha = 0.4)
            axes[1].set_aspect(1.0)

            axes[2].scatter(*end_conditioned_reverse_samples.T)
            for example_conditioned_reverse_trajectory in example_conditioned_reverse_trajectories:
                axes[2].plot(*example_conditioned_reverse_trajectory.T, alpha = 0.4)
            axes[2].set_aspect(1.0)

            axes[3].scatter(*end_forward_samples.T)
            for example_forward_trajectory in example_forward_trajectories:
                axes[3].plot(*example_forward_trajectory.T, alpha = 0.4)
            axes[3].set_aspect(1.0)

            axes[4].plot(all_losses_smoothed)
            
            fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/z_approx_scorematching/schedule.png')
