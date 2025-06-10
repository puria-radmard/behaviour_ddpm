import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Import from your moment prediction script
from ddpm.dynamical.connectivity_distn import PointMassConnectivityDistribution, DiagonalGaussianConnectivityDistribution
from ddpm.dynamical.moment_pred import plot_ellipse
from ddpm.dynamical.vi_fitting_utils import calculate_elbo_temporal, generate_data_inner, generate_target_data

if __name__ == '__main__':
    
    # Hyperparameters
    from purias_utils.util.arguments_yaml import ConfigNamepace
    args = ConfigNamepace.from_yaml_path('ddpm/dynamical/args.yaml')

    kl_beta = 0.0    # Infinite data -> no KL
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Sigma_xi = torch.eye(args.neurons).to(device) * args.sigma2_xi
    
    # Initialize actual connectivity distributions
    # For now, these are placeholders that will error if called
    prior_dist = DiagonalGaussianConnectivityDistribution(args.neurons, 1.0).to(device)
    variational_dist = DiagonalGaussianConnectivityDistribution(args.neurons, 1.0).to(device)

    W_star = torch.randn(args.neurons, args.neurons, device=device)
    eigvals = torch.linalg.eigvals(W_star)
    spectral_radius = eigvals.abs().max()
    W_star = W_star / (1.1 * spectral_radius)
    target_dist = PointMassConnectivityDistribution(args.neurons, W_star, eps=0.01).to(device)
    for param in target_dist.parameters():
        param.requires_grad = False
    target_dist.eval()

    optimizer = torch.optim.Adam(params=variational_dist.parameters(), lr = 0.01)
    kls = []
    llhs = []
    max_llhs = []
    true_posts = []
    
    # Training loop
    for epoch in tqdm(range(args.num_epochs)):
        
        # Sample batch of inputs h ~ p(h)
        h_scales = torch.ones(args.batch_size, args.neurons, device=device)
        h_scales.exponential_(lambd = 1 / float(args.h_scale))
        h_batch = torch.randn_like(h_scales) * h_scales
        
        # Generate target data u* ~ p*(u | W*, h) [batch, trials (1), time, neuron]
        with torch.no_grad():
            u_target = generate_target_data(
                target_dist, h_batch, args.timesteps, args.tau, args.dt, args.k, args.n, Sigma_xi, True, device
            )
        
        # Compute ELBO and gradients
        elbo, llh, kl = calculate_elbo_temporal(
            variational_dist, prior_dist, h_batch, u_target, 
            args.K, args.tau, args.dt, args.k, args.n, Sigma_xi, kl_beta, device
        )

        with torch.no_grad():
            max_elbo, max_llh, _ = calculate_elbo_temporal(
                target_dist, prior_dist, h_batch, u_target, 
                args.K, args.tau, args.dt, args.k, args.n, Sigma_xi, 0.0, device
            )
        
        # Backward pass (assuming variational_dist has parameters)
        loss = -elbo  # Maximize ELBO = minimize -ELBO
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        kls.append(kl.item())
        llhs.append(llh.item())
        max_llhs.append(max_llh.item())

        true_post = variational_dist.log_likelihood(W_star)
        true_posts.append(true_post.item())
        
        print(f'Epoch {epoch}, ELBO: {llh.item():.4f}, Max ELBO {max_llh.item():.4f}, log q(W*) = {true_post.item():.4f}')

        # Plot loss curve and save
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        axes[0].plot(llhs)
        axes[0].plot(max_llhs, linestyle = '--', color = 'black')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('ELBO')
        axes[1].plot(kls)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('KL')
        axes[2].plot(true_posts)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('$\log q(W^*)$')
        fig.savefig('ddpm/dynamical/logs/vi_fitting_temporal/vi_training_temporal.png')
        plt.close(fig)

        # Save model parameters
        torch.save(target_dist.state_dict(), 'ddpm/dynamical/logs/vi_fitting_temporal/target_dist_temporal.mdl')
        torch.save(variational_dist.state_dict(), 'ddpm/dynamical/logs/vi_fitting_temporal/variational_dist_temporal.mdl')

        if epoch % 10 == 0:

            with torch.no_grad():

                # Plot target and variational mean dynamics
                start_u = torch.zeros(args.batch_size, args.K, args.neurons, device=device)

                variational_dist_mean = variational_dist.mean.data[None,None].repeat(1, args.K, 1, 1)   # [B, 1, N, N]
                stacked_u_variational_mean = generate_data_inner(    # [B, args.K, T, N]
                    variational_dist_mean, h_batch, args.timesteps, args.tau, args.dt, args.k, args.n, Sigma_xi, True, device,
                    start_u = start_u
                )
                mean_u_variational_mean = stacked_u_variational_mean.mean(1).cpu()  # [B, T, N]
                # Covariance of final activity over trials: [B, T, N, N]
                cov_u_variational_mean = torch.stack([
                    torch.cov(stacked_u_variational_mean[b, :, t, :].T) 
                    for b in range(args.batch_size) 
                    for t in range(args.timesteps)
                ]).view(args.batch_size, args.timesteps, args.neurons, args.neurons).cpu()

                # Plot target and variational mean dynamics
                target_dist_mean = target_dist.mean.data[None,None].repeat(1, args.K, 1, 1)   # [B, 1, N, N]
                stacked_u_target_mean = generate_data_inner(    # [B, args.K, T, N]
                    target_dist_mean, h_batch, args.timesteps, args.tau, args.dt, args.k, args.n, Sigma_xi, True, device,
                    start_u = start_u
                )
                mean_u_target_mean = stacked_u_target_mean.mean(1).cpu()   # [B, T, N]
                # Covariance of final activity over trials: [B, T, N, N]
                cov_u_target_mean = torch.stack([
                    torch.cov(stacked_u_target_mean[b, :, t, :].T) 
                    for b in range(args.batch_size) 
                    for t in range(args.timesteps)
                ]).view(args.batch_size, args.timesteps, args.neurons, args.neurons).cpu()

                # Determine number of columns and figure size
                n_cols = 3 if args.neurons == 2 else 2
                fig, axes = plt.subplots(args.batch_size, n_cols, figsize=(5 * n_cols, 5 * args.batch_size))
                if args.batch_size == 1:
                    axes = axes.reshape(1, -1)

                axes[0, 0].set_title('Mean trajectories (solid=variational, dashed=target)')
                axes[0, 1].set_title('Final State Comparison')
                    
                for b in range(args.batch_size):
                    # Column 1: Time series comparison
                    for neuron in range(args.neurons):
                        var_col = axes[b, 0].plot(mean_u_variational_mean[b, :, neuron], 
                                    label=f'Neuron {neuron} (variational mean)', linestyle='-')[0].get_color()
                        axes[b, 0].plot(mean_u_target_mean[b, :, neuron], 
                                    label=f'Neuron {neuron} (target mean)', linestyle='--', color = var_col)
                    axes[b, 0].set_xlabel('Timestep')
                    axes[b, 0].set_ylabel('Mean $u$ over trials')
                    # axes[b, 0].legend()
                    
                    # Column 2: Final state comparison
                    axes[b, 1].scatter(mean_u_target_mean[b, -1], mean_u_variational_mean[b, -1])
                    axes[b, 1].plot([mean_u_target_mean[b, -1].min(), mean_u_target_mean[b, -1].max()], 
                                [mean_u_target_mean[b, -1].min(), mean_u_target_mean[b, -1].max()], 
                                '--', alpha=0.5)
                    axes[b, 1].set_xlabel('Target mean (final)')
                    axes[b, 1].set_ylabel('Variational mean (final)')
                        
                    
                    # Column 3: 2D phase space visualization (only for args.neurons == 2)
                    if args.neurons == 2:
                        # Final samples from variational distribution (blue points)
                        final_u_variational = stacked_u_variational_mean[b, :, -1, :].cpu()  # [args.K, 2]
                        axes[b, 2].scatter(final_u_variational[:, 0], final_u_variational[:, 1], 
                                        c='blue', alpha=0.6, s=20, label='Variational samples')
                        
                        # Final samples from target distribution (red points) 
                        final_u_target = stacked_u_target_mean[b, :, -1, :].cpu()  # [args.K, 2]
                        axes[b, 2].scatter(final_u_target[:, 0], final_u_target[:, 1], 
                                        c='red', alpha=0.6, s=20, label='Target samples')
                        
                        # Variational mean and covariance (blue)
                        variational_mean_final = mean_u_variational_mean[b, -1, :]  # [2]
                        variational_cov_final = cov_u_variational_mean[b, -1, :, :]  # [2, 2]
                        plot_ellipse(axes[b, 2], variational_mean_final, variational_cov_final, 
                                    'blue', alpha=0.2, label='Variational (mean & 2σ)')
                        
                        # Target mean and covariance (red)
                        target_mean_final = mean_u_target_mean[b, -1, :]  # [2]
                        target_cov_final = cov_u_target_mean[b, -1, :, :]  # [2, 2]
                        plot_ellipse(axes[b, 2], target_mean_final, target_cov_final, 
                                    'red', alpha=0.2, label='Target (mean & 2σ)')
                        
                        axes[b, 2].set_xlabel('Neuron 0')
                        axes[b, 2].set_ylabel('Neuron 1') 
                        axes[b, 2].set_title(f'Batch {b}: 2D Phase Space (Final State)')
                        axes[b, 2].legend()
                        axes[b, 2].grid(True, alpha=0.3)
                        axes[b, 2].set_aspect('equal')

                        axes[b, 2].sharey(axes[b-1, 2])
                        axes[b, 2].sharex(axes[b-1, 2])

                # fig.tight_layout()
                fig.savefig('ddpm/dynamical/logs/vi_fitting_temporal/vi_training_examples_temporal.png')
                plt.close(fig)

    
    print("Training complete!")