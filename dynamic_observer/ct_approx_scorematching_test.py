from __future__ import annotations

import torch

from dynamic_observer.model import *


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    noise_schedule = LinearIncreaseNoiseSchedule(0.2, 0.2, duration = 50)

    score_func = FCScoreApproximator(
        sample_size = 2,
        hidden_layers = [16, 16],
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
    num_reverse_dynamics_steps = 100

    target_m0 = torch.tensor([-50.0, -10.0])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1)
    target_S0 = torch.tensor([[10.0, -3.0], [-3.0, 4.0]])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1, 1)
    vector_stimulus = (target_m0, target_S0)

    base_samples = torch.randn(batch_size, 2)
    all_reverse_trajectories = diffmodel.run_unconditioned_reverse_dynamics(base_samples, vector_stimulus, num_reverse_dynamics_steps).cpu().numpy()
    example_reverse_trajectories = all_reverse_trajectories[:5]
    end_reverse_samples = all_reverse_trajectories[:,-1,:]

    
    observations = 30.0 + torch.zeros(num_reverse_dynamics_steps, batch_size, 1)  # y
    projection_matrix = torch.tensor([[0.1961, -0.9806]])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1, 1)  # A
    observation_noise_covar = torch.tensor([[1.0, 0.0], [0.0, 1.0]])[None,None].repeat(num_reverse_dynamics_steps, batch_size, 1, 1)

    base_samples = torch.randn(batch_size, 2)
    all_conditioned_reverse_trajectories = diffmodel.run_conditioned_reverse_dynamics(
        base_samples, vector_stimulus, num_reverse_dynamics_steps, observations, projection_matrix, observation_noise_covar
    ).cpu().numpy()
    example_conditioned_reverse_trajectories = all_conditioned_reverse_trajectories[:5]
    end_conditioned_reverse_samples = all_conditioned_reverse_trajectories[:,-1,:]
    
    real_samples = torch.randn(batch_size, 2) + target_m0[0,0]      # Not quite real samples!
    all_forward_trajectories = diffmodel.run_forward_dynamics(real_samples, 100).cpu().numpy()
    example_forward_trajectories = all_forward_trajectories[:5]
    end_forward_samples = all_forward_trajectories[:,-1,:]

    fig, axes = plt.subplots(1, 4, figsize = (20, 5))
    scaling_factor_time, scaling_factor = noise_schedule.summarise_noising_factor(100)
    axes[0].plot(scaling_factor_time.cpu().numpy(), scaling_factor.cpu().numpy())
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
    
    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/z_exact_scorematching/schedule.png')
