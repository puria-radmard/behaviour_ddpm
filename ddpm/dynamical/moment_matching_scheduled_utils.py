from ddpm.dynamical.moment_pred import *


def generate_empirical_moments_batch(W_target, h_batch, args, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi_chol, Sigma_xi_chols):
    """Generate empirical moments for a batch of input conditions"""
    batch_size = h_batch.shape[0]
    trials_target = args.K
    num_neurons = W_target.shape[-1]
    device = h_batch.device
    
    # Run dynamics: [batch, trials, neurons] format
    u_target = torch.randn(batch_size, trials_target, num_neurons, device=device)
    
    # Use pre-computed values
    Sigma_xi_chol = initial_Sigma_xi_chol
    h_current = h_batch.clone()  # [batch, neurons]
    
    for t in tqdm(range(timesteps), desc="Generating target moments", disable=True):
        
        if t >= timesteps_stim:
            if timesteps_no_stim > 0:
                idx = t - timesteps_stim
                Sigma_xi_chol = Sigma_xi_chols[idx]
            h_current = h_current * 0.0
        
        # Generate noise: [batch, trials, neurons]
        dxi = torch.randn(batch_size, trials_target, num_neurons, device=device) @ Sigma_xi_chol.T * math.sqrt(args.dt)
        
        # u dynamics with target weights - use broadcasting instead of expansion
        du = du_dt(u_target, h_current, args.tau, W_target[None,None], args.dt, dxi, args.k, args.n)
        u_target = u_target + du * args.dt
    
    # Compute empirical moments from final state: [batch, trials, neurons]
    empirical_mean = u_target.mean(dim=1)  # [batch, neurons] - average over trials
    
    # Compute covariance for each batch item
    empirical_cov = torch.zeros(batch_size, num_neurons, num_neurons, device=device)
    for b in range(batch_size):
        # Center the data
        centered = u_target[b] - empirical_mean[b].unsqueeze(0)  # [trials, neurons]
        empirical_cov[b] = (centered.T @ centered) / (trials_target - 1)  # [neurons, neurons]
    
    return empirical_mean, empirical_cov, u_target

def predict_moments(W_learnable, h_batch, args, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi, Sigma_xis):
    """Generate moment predictions from learnable network"""
    batch_size = h_batch.shape[0]
    num_neurons = W_learnable.shape[-1]
    device = h_batch.device
    
    # Reset moment predictions
    trials_pred = 1  # For theoretical predictions
    mu = torch.zeros(batch_size, trials_pred, num_neurons, device=device)
    Sigma = torch.eye(num_neurons, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, trials_pred, 1, 1)
    
    # Just use the base weight matrix, broadcasting will handle the rest
    W_pred = W_learnable[None, None]
    
    is_linear = (args.n == -1)
    
    if is_linear:
        gamma_val = gamma(args.n, args.k, mu, Sigma)
        J_val = J(W_pred, args.tau, gamma_val)
    
    # Reset input and noise schedule
    h_current = h_batch.clone()
    Sigma_xi = initial_Sigma_xi
    
    # Run moment prediction dynamics
    for t in range(timesteps):
        
        if t >= timesteps_stim:
            if timesteps_no_stim > 0:
                idx = t - timesteps_stim
                Sigma_xi = Sigma_xis[idx]
            h_current = h_current * 0.0
        
        # Moment dynamics
        if is_linear:
            nu_val = nu(args.n, args.k, mu, Sigma)
            dSigma = dSigma0_dt(Sigma_xi, J_val, Sigma, args.tau)
            dmu = dmu_dt(mu, h_current, W_pred, nu_val, args.tau)
        else:
            nu_val = nu(args.n, args.k, mu, Sigma)
            gamma_val = gamma(args.n, args.k, mu, Sigma)
            J_val = J(W_pred, args.tau, gamma_val)
            dSigma = dSigma0_dt(Sigma_xi, J_val, Sigma, args.tau)
            dmu = dmu_dt(mu, h_current, W_pred, nu_val, args.tau)
        
        Sigma = Sigma + dSigma * args.dt
        mu = mu + dmu * args.dt
    
    # Extract final predicted moments
    pred_mean = mu[:, 0, :]  # [batch, neurons]
    pred_cov = Sigma[:, 0, :, :]  # [batch, neurons, neurons]
    
    return pred_mean, pred_cov



def generate_learned_traj(W_learnable, h_single, args, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi_chol, Sigma_xi_chols):
    """Generate outputs from learned network for comparison"""
    trials_comparison = min(args.K, 1000)  # Use fewer trials for visualization
    num_neurons = W_learnable.shape[-1]
    device = h_single.device
    
    u_learned = torch.randn(1, trials_comparison, num_neurons, device=device)
    
    # Use pre-computed constants
    Sigma_xi_chol = initial_Sigma_xi_chol
    h_current = h_single.unsqueeze(0)  # [1, neurons]
    
    for t in range(timesteps):
        
        if t >= timesteps_stim:
            if timesteps_no_stim > 0:
                idx = t - timesteps_stim
                Sigma_xi_chol = Sigma_xi_chols[idx]
            h_current = h_current * 0.0
        
        dxi = torch.randn(1, trials_comparison, num_neurons, device=device) @ Sigma_xi_chol.T * math.sqrt(args.dt)
        
        # Use learned W - broadcasting will handle dimensions
        du = du_dt(u_learned, h_current, args.tau, W_learnable[None, None], args.dt, dxi, args.k, args.n)
        u_learned = u_learned + du * args.dt
    
    return u_learned[0]


def generate_learned_outputs(W_learnable, h_single, args, output_weights, output_bias, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi_chol, Sigma_xi_chols):
    
    final_states = generate_learned_traj(W_learnable, h_single, args, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi_chol, Sigma_xi_chols)
    
    # Generate final outputs
    final_activities = f(final_states, args.k, args.n)  # [trials, neurons]
    final_outputs = (final_activities @ output_weights.T) + output_bias.unsqueeze(0)
    
    return final_outputs  # Return [trials, output_dim]
