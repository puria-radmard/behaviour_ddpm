import torch
from typing import Tuple
import math
from torch import Tensor as _T

# Import from your moment prediction script
from ddpm.dynamical.connectivity_distn import ConnectivityDistribution
from ddpm.dynamical.moment_pred import (
    du_dt, dmu_dt, nu, dSigma0_dt, J, gamma,
)

from ddpm.dynamical.moment_pred import du_dt_deterministic


def generate_data_inner(
    W: _T, # [batch, trials, neurons, neurons]
    h_batch: _T,  # [batch, neurons]
    timesteps: int,
    tau: float,
    dt: float,
    k: float,
    n: int,
    Sigma_xi: _T,  # [neurons, neurons]
    full_traj: bool,
    device: torch.device,
    *_,
    start_u: _T = None  # [batch, trials, neurons]
) -> _T:
    """    
    Returns: u* of shape [batch, neurons] (final values) or [batch, time, neurons]
    """
    batch, neurons = h_batch.shape
    num_trials = W.shape[1]
    _expected_shape_1 = (batch, num_trials, neurons, neurons)
    _expected_shape_2 = (1, num_trials, neurons, neurons)
    if (tuple(W.shape) != _expected_shape_1) and (tuple(W.shape) != _expected_shape_2):
        raise ValueError(f"Expected W of shape {_expected_shape_1} or {_expected_shape_2}, but got {tuple(W.shape)}")
    
    # Initialize dynamics
    if start_u is None:
        u = torch.randn(batch, num_trials, neurons, device=device)
    else:
        assert tuple(start_u.shape) == (batch, num_trials, neurons)
        u = start_u.to(device=device)
    Sigma_xi_chol = torch.linalg.cholesky(Sigma_xi)

    if full_traj:
        us = []
    
    # Run dynamics to steady state
    for t in range(timesteps):
        dxi = torch.randn(batch, num_trials, neurons, device=device) @ Sigma_xi_chol.T * math.sqrt(dt)
        du = du_dt(u, h_batch, tau, W, dt, dxi, k, n)
        u = u + du * dt

        if full_traj:
            us.append(u.clone())

    if full_traj:
        return torch.stack(us, 2)  # [batch, trials, time, neurons]
    
    else:
        return u  # [batch, trials, neurons]


def generate_target_data(
    target_dist: ConnectivityDistribution,
    h_batch: _T,  # [batch, neurons]
    timesteps: int,
    tau: float,
    dt: float,
    k: float,
    n: int,
    Sigma_xi: _T,  # [neurons, neurons]
    full_traj: bool,
    device: torch.device,
) -> _T:
    """Generate target neural activity u* from true distribution p*(W)
    
    Returns:
    if 
        u* of shape [batch, neurons] (final values ~= steady-state, trials squeezed out)
    """
    W_target = target_dist.sample(1, 1).to(device)
    
    ret = generate_data_inner(
        W_target, h_batch, timesteps, tau, dt, k, n, Sigma_xi, full_traj, device
    )

    return ret.squeeze(1) if not full_traj else ret


def compute_moments_trajectory(
    W: _T,  # [batch, trials, neurons, neurons]
    h_batch: _T,  # [batch, neurons]
    timesteps: int,
    tau: float,
    dt: float,
    k: float,
    n: int,
    Sigma_xi: _T,  # [neurons, neurons]
    device: torch.device,
) -> Tuple[_T, _T]:
    """Compute theoretical steady-state moments μ, Σ
    
    Returns: 
        mu: [batch, trials, time, neurons]
        Sigma: [batch, trials, time, neurons, neurons]
    """
    batch, trials, neurons, _ = W.shape
    
    # Initialize moments
    mu = torch.zeros(batch, trials, neurons, device=device)
    Sigma = Sigma_xi.unsqueeze(0).unsqueeze(0).repeat(batch, trials, 1, 1).to(device)

    mus = []
    Sigmas = []
    
    # Integrate moment equations to steady state
    for t in range(timesteps):
        nu_val = nu(n, k, mu, Sigma)
        gamma_val = gamma(n, k, mu, Sigma)
        J_val = J(W, tau, gamma_val)
        
        dSigma = dSigma0_dt(Sigma_xi, J_val, Sigma)
        dmu = dmu_dt(mu, h_batch, W, nu_val, tau)
        
        Sigma = Sigma + dSigma * dt
        #if (torch.diagonal(Sigma, 0, -1, -2) < 0).any():
        #    import pdb; pdb.set_trace()
        mu = mu + dmu * dt

        mus.append(mu.clone())
        Sigmas.append(Sigma.clone())

    mus = torch.stack(mus, 2)
    Sigmas = torch.stack(Sigmas, 2)
    
    return mus, Sigmas


def compute_log_likelihood(
    u_target: _T,  # [batch, neurons] 
    mu_pred: _T,   # [batch, trials, neurons]
    Sigma_pred: _T,  # [batch, trials, neurons, neurons]
    device: torch.device,
) -> _T:
    """Compute log p(u_target | mu_pred, Sigma_pred) averaged over trials
    
    Returns: scalar log likelihood
    """
    batch, trials, neurons = mu_pred.shape
    
    # Expand u_target to match trials dimension
    u_target_expanded = u_target.unsqueeze(1).expand(-1, trials, -1)  # [batch, trials, neurons]
    
    # Compute log likelihood for each (batch, trial) pair
    log_probs = []
    for b in range(batch):
        for t in range(trials):
            mu_bt = mu_pred[b, t]  # [neurons]
            Sigma_bt = Sigma_pred[b, t]  # [neurons, neurons]
            u_bt = u_target_expanded[b, t]  # [neurons]
            
            # Add small regularization to covariance for numerical stability
            Sigma_reg = Sigma_bt + 1e-5 * torch.eye(neurons, device=device)
            
            # Multivariate normal log probability
            dist = torch.distributions.MultivariateNormal(mu_bt, Sigma_reg)
            log_prob = dist.log_prob(u_bt)
            log_probs.append(log_prob)
    
    # Average over all (batch, trials) pairs
    return torch.stack(log_probs).mean()


def calculate_elbo(
    variational_dist: ConnectivityDistribution,
    prior_dist: ConnectivityDistribution,
    h_batch: _T,  # [batch, neurons]
    u_target: _T,  # [batch, neurons]
    K: int,  # Number of samples from q(W)
    timesteps: int,
    tau: float,
    dt: float,
    k: float,
    n: int,
    Sigma_xi: _T,
    kl_beta: float,
    device: torch.device,
) -> _T:
    """Compute ELBO and take gradient step
    
    Returns: ELBO value
    """
    batch, neurons = h_batch.shape
    
    # Sample from variational distribution: [batch, K, neurons, neurons]
    W_samples = variational_dist.sample(batch, K).to(device)
    
    # Compute theoretical moments for each sample
    mu_pred_all, Sigma_pred_all = compute_moments_trajectory(
        W_samples, h_batch, timesteps, tau, dt, k, n, Sigma_xi, device
    )
    # [batch, trials, neurons], [batch, trials, neurons, neurons]
    mu_pred = mu_pred_all[:,:,-1]
    Sigma_pred = Sigma_pred_all[:,:,-1]
    Sigma_pred = 1e-3 * torch.eye(neurons, device=device)[None,None]
    
    # Compute expected log likelihood - [batch, MC trials]
    log_likelihood_obj = torch.distributions.MultivariateNormal(loc=mu_pred, covariance_matrix=Sigma_pred)
    log_likelihoods = log_likelihood_obj.log_prob(u_target.unsqueeze(1))

    # ELBO = E[log p(u|W)] - KL[q(W)||p(W)]
    # KL term may be removed in infinite data case!
    total_llh = log_likelihoods.mean(1).sum()
    elbo = total_llh
    if kl_beta > 0:
        kl_div = variational_dist.kl_against(prior_dist)
        elbo = elbo - kl_beta * kl_div
    else:
        kl_div = torch.tensor(torch.nan)
    
    return elbo, total_llh, kl_div





def calculate_elbo_temporal(
    variational_dist: ConnectivityDistribution,
    prior_dist: ConnectivityDistribution,
    h_batch: _T,  # [batch, neurons]
    u_target: _T,  # [batch, time, neurons]
    K: int,  # Number of samples from q(W)
    tau: float,
    dt: float,
    k: float,
    n: int,
    Sigma_xi: _T,
    kl_beta: float,
    device: torch.device,
) -> _T:
    """Compute ELBO and take gradient step
    
    Returns: ELBO value
    """
    batch, neurons = h_batch.shape
    
    # Sample from variational distribution: [batch, K, neurons, neurons]
    W_samples = variational_dist.sample(batch, K).to(device)
    
    # Compute log likelihood of shape [batch, MC samples, time - 1]
    du_dt = du_dt_deterministic(    # [batch, MC samples, time - 1, neurons]
        u = u_target[:,:,:-1],
        h = h_batch,
        tau = tau,
        W = W_samples,
        k = k,
        n = n,
        parallelise_u=True
    )
    one_step_mean = u_target[:,:,:-1] + (du_dt * dt)
    log_likelihood_obj = torch.distributions.MultivariateNormal(loc=one_step_mean, covariance_matrix=dt * Sigma_xi)
    log_likelihoods = log_likelihood_obj.log_prob(u_target[:,:,1:])
    
    # ELBO = E[log p(u|W)] - KL[q(W)||p(W)]
    # KL term may be removed in infinite data case!
    total_llh = log_likelihoods.mean(1).sum()
    elbo = total_llh
    if kl_beta > 0:
        kl_div = variational_dist.kl_against(prior_dist)
        elbo = elbo - kl_beta * kl_div
    else:
        kl_div = torch.tensor(torch.nan)
    
    return elbo, total_llh, kl_div


