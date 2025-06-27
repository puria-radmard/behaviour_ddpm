from ddpm.dynamical.moment_pred import *


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    from purias_utils.util.arguments_yaml import ConfigNamepace
    args = ConfigNamepace.from_yaml_path('ddpm/dynamical/args_scheduled.yaml')

    # Generate stable W matrices for each batch item
    # W: [batch, trials, neurons, neurons]
    if 'load_path' in args.dict:
        sd = torch.load(args.load_path)

        W = sd['W'].unsqueeze(0).unsqueeze(0).cpu()
        num_neurons = 32
        batch_size = 1
        trials_target = 1
        save_suffix = "_loaded"

        output_weights = sd['out_weight.weight'].cpu()
        output_bias = sd['out_weight.bias'].cpu()

    else:
        batch_size = args.batch_size
        num_neurons = args.neurons
        trials_target = 1
        save_suffix = ""

        W = torch.randn(batch_size, trials_target, num_neurons, num_neurons)
        
        # Make each W stable by scaling spectral radius
        for b in range(batch_size):
            eigvals = torch.linalg.eigvals(W[b, 0])
            spectral_radius = eigvals.abs().max()
            scale = torch.rand(1) / 0.4 + 1.0
            W[b, 0] = W[b, 0] / (scale * spectral_radius)
        
    # Initial conditions
    # u: [batch, trials, neurons]
    u = torch.randn(batch_size, args.K, num_neurons)
    # h: [batch, neurons] - different input for each batch item
    if args.h_scale > 0.0:
        h_scales = torch.ones(batch_size, num_neurons)
        h_scales.exponential_(lambd = 1 / float(args.h_scale))
        h_batch = torch.randn_like(h_scales) * h_scales
    else:
        h_batch = torch.zeros(batch_size, num_neurons).float()

    us = [u.clone()]

    # Initialize mu, Sigma, and history
    # mu: [batch, trials, neurons]
    mu = torch.zeros(batch_size, trials_target, num_neurons)
    # Sigma: [batch, trials, neurons, neurons]
    Sigma = torch.eye(num_neurons).unsqueeze(0).unsqueeze(0).repeat(batch_size, trials_target, 1, 1)
    mus = [mu.clone()]

    is_linear = (args.n == -1)

    if is_linear:
        # For linear case, we can compute J once since gamma is constant
        gamma_val = gamma(args.n, args.k, mu, Sigma)  # [batch, trials, neurons]
        J_val = J(W, args.tau, gamma_val)  # [batch, trials, neurons, neurons]

    timesteps_stim = int(args.duration / args.dt)
    timesteps_no_stim = int(args.duration_no_stim / args.dt)
    timesteps = timesteps_stim + timesteps_no_stim

    scheduled_sigma2_xi = args.sigma2_xi_preschedule
    Sigma_xi = torch.eye(num_neurons) * scheduled_sigma2_xi  # [neurons, neurons]
    Sigma_xi_chol = Sigma_xi.sqrt()
    
    sigma2_xis = torch.linspace(args.sigma2_xi_start, args.sigma2_xi_end, timesteps_no_stim)

    for t in tqdm(range(timesteps)):
        
        if t >= timesteps_stim:
            scheduled_sigma2_xi = sigma2_xis[t - timesteps_stim]
            Sigma_xi = torch.eye(num_neurons) * scheduled_sigma2_xi  # [neurons, neurons]
            Sigma_xi_chol = Sigma_xi.sqrt()

            h_batch = h_batch * 0.0

        # Generate noise: [batch, trials, neurons]
        dxi = torch.randn(batch_size, args.K, num_neurons) @ Sigma_xi_chol.T * math.sqrt(args.dt)

        # u dynamics (per trial)
        du = du_dt(u, h_batch, args.tau, W, args.dt, dxi, args.k, args.n)
        u = u + du * args.dt
        us.append(u.clone())

        # mu/Sigma dynamics
        if is_linear:
            # Linear case: nu and gamma don't depend on current state
            nu_val = nu(args.n, args.k, mu, Sigma)  # [batch, trials, neurons]
            # J_val already computed above and is constant
            dSigma = dSigma0_dt(Sigma_xi, J_val, Sigma, args.tau)
            dmu = dmu_dt(mu, h_batch, W, nu_val, args.tau)  # [batch, trials, neurons]
        else:
            # Nonlinear case: recompute everything
            nu_val = nu(args.n, args.k, mu, Sigma)  # [batch, trials, neurons]
            gamma_val = gamma(args.n, args.k, mu, Sigma)  # [batch, trials, neurons]
            J_val = J(W, args.tau, gamma_val)  # [batch, trials, neurons, neurons]
            dSigma = dSigma0_dt(Sigma_xi, J_val, Sigma, args.tau)
            dmu = dmu_dt(mu, h_batch, W, nu_val, args.tau)  # [batch, trials, neurons]

        Sigma = Sigma + dSigma * args.dt
        mu = mu + dmu * args.dt
        mus.append(mu.clone())

    # Stack history: [batch, trials, timesteps+1, neurons]
    us_stacked = torch.stack(us, dim=2)

    # Since trials=1 for target generation, squeeze out trials dimension for plotting
    # [batch, timesteps+1, neurons]
    mean_us_stacked = us_stacked.mean(1)
    mus_stacked = torch.stack(mus, dim=2).squeeze(1)  # [batch, timesteps+1, neurons]

    # Plot for each batch item
    n_cols = 3 if num_neurons == 2 or 'load_path' in args.dict else 2
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(5 * n_cols, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for b in range(batch_size):
        # Column 1: Time series (shared)
        plot_time_series(axes[b, 0], mean_us_stacked, mus_stacked, b, num_neurons)
        
        # Column 2: Final state comparison (shared)
        plot_final_state_comparison(axes[b, 1], mean_us_stacked, mus_stacked, b)
        
        # Column 3: 2D phase space visualization (only for neurons == 2)
        if num_neurons == 2:
            # Final samples from u (red points)
            final_u_samples = us_stacked[b, :, -1, :]  # [trials, 2]
            axes[b, 2].scatter(final_u_samples[:, 0], final_u_samples[:, 1], 
                            c='red', alpha=0.6, s=20, label='Final u samples')
            
            # Empirical mean and covariance (red)
            empirical_mean = final_u_samples.mean(0)  # [2]
            empirical_cov = torch.cov(final_u_samples.T)  # [2, 2]
            plot_ellipse(axes[b, 2], empirical_mean, empirical_cov, 
                        'red', alpha=0.2, label='Empirical (mean & 2σ)')
            
            # Theoretical mean and covariance (blue)
            theoretical_mean = mus_stacked[b, -1, :]  # [2]
            theoretical_cov = Sigma[b, 0, :, :]  # [2, 2] (squeeze trials dimension)
            plot_ellipse(axes[b, 2], theoretical_mean, theoretical_cov, 
                        'blue', alpha=0.2, label='Theoretical (mean & 2σ)')
            
            axes[b, 2].set_xlabel('Neuron 0')
            axes[b, 2].set_ylabel('Neuron 1') 
            axes[b, 2].set_title(f'Batch {b}: 2D Phase Space (Final State)')
            axes[b, 2].legend()
            axes[b, 2].grid(True, alpha=0.3)
            axes[b, 2].set_aspect('equal', adjustable='box')

    if 'load_path' in args.dict:
        network_final_outputs = (f(us[-1], args.k, args.n) @ output_weights.T).squeeze(0) + output_bias.unsqueeze(0)
        axes[0, 2].scatter(*network_final_outputs.numpy().T)
        axes[0, 2].set_title('Final output of trained network across trials')
        
    if batch_size > 1:
        fig.tight_layout()
    fig_path = f'ddpm/dynamical/logs/moment_pred_scheduled/u_history_N{num_neurons}_n{args.n}{save_suffix}.png'
    print(fig_path)
    plt.savefig(fig_path)
    plt.close()
    