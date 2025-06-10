import torch
from torch import Tensor as _T
import math
from tqdm import tqdm
from typing import Optional


def pdf(x: _T) -> _T:
    return (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x ** 2)

def cdf(x: _T) -> _T:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def f(u: _T, k: float, n: int) -> _T:
    if n == -1:
        return k * u
    else:
        return k * torch.pow(torch.relu(u), n)


def du_dt_deterministic(u: _T, h: _T, tau: float, W: _T, k: float, n: int, *_, parallelise_u: bool = False) -> _T:
    # u [batch, trials, time, neurons] if parallelise_u else [batch, trials, neurons]
    # W: [batch, trials, neurons, neurons]
    # h: [batch, neurons]
    r = f(u, k, n)  # [batch, trials, neurons]
    
    # Einstein sum: batch,trials,neurons @ batch,trials,neurons,neurons -> batch,trials,neurons
    if parallelise_u:
        Wr = torch.einsum('bkij,bktj->bkti', W, r)  # [batch, trials, time, neurons]
        return ((-u + h.unsqueeze(1).unsqueeze(1) + Wr) / tau)
    else:
        Wr = torch.einsum('bkij,bkj->bki', W, r)  # [batch, trials, neurons]
        return ((-u + h.unsqueeze(1) + Wr) / tau)

def du_dt(u: _T, h: _T, tau: float, W: _T, dt: float, dxi: _T, k: float, n: int) -> _T:
    # u, dxi: [batch, trials, neurons]
    # W: [batch, trials, neurons, neurons]
    # h: [batch, neurons]
    return du_dt_deterministic(u,h,tau,W,k,n) + (dxi / dt)

def dmu_dt(mu: _T, h: _T, W: _T, nu: _T, tau: float) -> _T:
    # mu, nu: [batch, trials, neurons]
    # W: [batch, trials, neurons, neurons] 
    # h: [batch, neurons]
    Wnu = torch.einsum('bkij,bkj->bki', W, nu)  # [batch, trials, neurons]
    ret = (-mu + h.unsqueeze(1) + Wnu) / tau
    return ret

def nu(n: int, k: float, mu: _T, Sigma: _T) -> _T:
    # mu: [batch, trials, neurons], Sigma: [batch, trials, neurons, neurons]
    if n == -1:
        return k * mu

    variance = torch.diagonal(Sigma, dim1=-2, dim2=-1)  # [batch, trials, neurons]
    if n == 1:
        stds = variance.sqrt()
        mu_over_std = mu / (stds + 1e-8)  # Add small epsilon for numerical stability
        ret = k * mu * cdf(mu_over_std) + k * stds * pdf(mu_over_std)
        return ret
    elif n == 2:
        stds = variance.sqrt()
        mu_over_std = mu / (stds + 1e-8)  # Add small epsilon for numerical stability
        return (mu * nu(1, k, mu, Sigma)) + (k * variance * cdf(mu_over_std))
    elif n > 2:
        return (mu * nu(n-1, k, mu, Sigma)) + ((n-1) * variance * nu(n-2, k, mu, Sigma))    # FIXME: extremely inefficient

def dSigma0_dt(Sigma_xi: _T, J: _T, Sigma0: _T, tau: float) -> _T:
    # Sigma_xi: [neurons, neurons]
    # J: [batch, trials, neurons, neurons]
    # Sigma0: [batch, trials, neurons, neurons]
    JT = J.transpose(-1, -2)  # [batch, trials, neurons, neurons]
    # Broadcast Sigma_xi to match batch and trials dimensions
    Sigma_xi_expanded = (Sigma_xi).unsqueeze(0).unsqueeze(0)  # [1, 1, neurons, neurons]
    return Sigma_xi_expanded + torch.matmul(J, Sigma0) + torch.matmul(Sigma0, JT)

def J(W: _T, tau: float, gammas: _T):
    # W: [batch, trials, neurons, neurons]
    # gammas: [batch, trials, neurons]
    batch, trials, num_neurons, _ = W.shape
    gammas_diag = torch.diag_embed(gammas)  # [batch, trials, neurons, neurons]
    Wg = torch.matmul(W, gammas_diag)  # [batch, trials, neurons, neurons]
    eye = torch.eye(num_neurons, device=W.device, dtype=W.dtype).unsqueeze(0).unsqueeze(0)
    return (Wg - eye) / tau

def gamma(n: int, k: float, mu: _T, Sigma: _T) -> _T:
    # mu: [batch, trials, neurons], Sigma: [batch, trials, neurons, neurons]
    if n == -1:  # Linear case
        return k * torch.ones_like(mu)  # gamma = k for linear case

    variance = torch.diagonal(Sigma, dim1=-2, dim2=-1)  # [batch, trials, neurons]
    if n == 1:
        stds = variance.sqrt()
        return k * cdf(mu / (stds + 1e-8))
    else:
        return n * nu(n - 1, k, mu, Sigma)


def plot_ellipse(ax, mean, cov, color, alpha=0.3, label=None):
    """Plot confidence ellipse for 2D Gaussian distribution"""
    from matplotlib.patches import Ellipse
    eigenvals, eigenvecs = torch.linalg.eigh(cov)
    eigenvals = eigenvals.abs()  # Ensure positive
    angle = torch.atan2(eigenvecs[1, 0], eigenvecs[0, 0]) * 180 / math.pi
    
    # 2-sigma ellipse
    width = 2 * torch.sqrt(eigenvals[0])
    height = 2 * torch.sqrt(eigenvals[1])
    
    ellipse = Ellipse(xy=(mean[0].item(), mean[1].item()), 
                     width=width.item(), height=height.item(),
                     angle=angle.item(), facecolor=color, alpha=alpha,
                     edgecolor=color, linewidth=2, label=label)
    ax.add_patch(ellipse)
    ax.scatter(mean[0].item(), mean[1].item(), color=color, s=100, zorder=5)

def plot_time_series(ax, mean_us_stacked, mus_stacked, batch_idx, neurons):
    """Plot time series comparison between empirical and theoretical means"""
    for neuron in range(neurons):
        line = ax.plot(mean_us_stacked[batch_idx, :, neuron])
                      #label=f'Neuron {neuron} (empirical)')
        ax.plot(mus_stacked[batch_idx, :, neuron], 
               linestyle='--', color=line[0].get_color(), alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean $u$')
    ax.set_title(f'Batch {batch_idx}: Mean $u$ over time (solid=empirical, dashed=theoretical)')
    ax.legend()

def plot_final_state_comparison(ax, mean_us_stacked, mus_stacked, batch_idx):
    """Plot final state empirical vs theoretical comparison"""
    ax.scatter(mean_us_stacked[batch_idx, -1], mus_stacked[batch_idx, -1])
    ax.set_xlabel('Empirical mean (target u*)')
    ax.set_ylabel('Theoretical mean (predicted μ)')
    ax.set_title(f'Batch {batch_idx}: Final State Comparison')


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    from purias_utils.util.arguments_yaml import ConfigNamepace
    args = ConfigNamepace.from_yaml_path('ddpm/dynamical/args.yaml')

    # Parameters
    trials_target = 1

    Sigma_xi = torch.eye(args.neurons) * args.sigma2_xi  # [neurons, neurons]
    Sigma_xi_chol = torch.cholesky(Sigma_xi)

    # Generate stable W matrices for each batch item
    # W: [batch, trials, neurons, neurons]
    W = torch.randn(args.batch_size, trials_target, args.neurons, args.neurons)
    
    # Make each W stable by scaling spectral radius
    for b in range(args.batch_size):
        eigvals = torch.linalg.eigvals(W[b, 0])
        spectral_radius = eigvals.abs().max()
        scale = torch.rand(1) / 0.4 + 1.0
        W[b, 0] = W[b, 0] / (scale * spectral_radius)
    
    # Initial conditions
    # u: [batch, trials, neurons]
    u = torch.randn(args.batch_size, args.K, args.neurons)
    # h: [batch, neurons] - different input for each batch item
    h_scales = torch.ones(args.batch_size, args.neurons)
    h_scales.exponential_(lambd = 1 / float(args.h_scale))
    h_batch = torch.randn_like(h_scales) * h_scales

    us = [u.clone()]

    # Initialize mu, Sigma, and history
    # mu: [batch, trials, neurons]
    mu = torch.zeros(args.batch_size, trials_target, args.neurons)
    # Sigma: [batch, trials, neurons, neurons]
    Sigma = torch.eye(args.neurons).unsqueeze(0).unsqueeze(0).repeat(args.batch_size, trials_target, 1, 1)
    mus = [mu.clone()]

    is_linear = (args.n == -1)

    if is_linear:
        # For linear case, we can compute J once since gamma is constant
        gamma_val = gamma(args.n, args.k, mu, Sigma)  # [batch, trials, neurons]
        J_val = J(W, args.tau, gamma_val)  # [batch, trials, neurons, neurons]

    timesteps = int(args.duration / args.dt)

    for t in tqdm(range(timesteps)):
        # Generate noise: [batch, trials, neurons]
        dxi = torch.randn(args.batch_size, args.K, args.neurons) @ Sigma_xi_chol.T * math.sqrt(args.dt)

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
    n_cols = 3 if args.neurons == 2 else 2
    fig, axes = plt.subplots(args.batch_size, n_cols, figsize=(5 * n_cols, 5 * args.batch_size))
    if args.batch_size == 1:
        axes = axes.reshape(1, -1)

    for b in range(args.batch_size):
        # Column 1: Time series (shared)
        plot_time_series(axes[b, 0], mean_us_stacked, mus_stacked, b, args.neurons)
        
        # Column 2: Final state comparison (shared)
        plot_final_state_comparison(axes[b, 1], mean_us_stacked, mus_stacked, b)
        
        # Column 3: 2D phase space visualization (only for neurons == 2)
        if args.neurons == 2:
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
        
    fig.tight_layout()
    fig_path = f'ddpm/dynamical/logs/moment_pred/u_history_N{args.neurons}_n{args.n}.png'
    print(fig_path)
    plt.savefig(fig_path)
    plt.close()
    