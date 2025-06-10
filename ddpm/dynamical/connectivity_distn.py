from __future__ import annotations

import torch, math
from torch import Tensor as _T, nn
from abc import ABC, abstractmethod

class ConnectivityDistribution(ABC, nn.Module):
    
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons
    
    @abstractmethod
    def sample(self, batches: int, trials: int) -> _T:
        """Sample connectivity matrices of shape [batches, trials, neurons, neurons]"""
        pass
    
    @abstractmethod 
    def kl_against(self, other: ConnectivityDistribution) -> _T:
        """Compute KL divergence KL[self || other]"""
        pass

    @abstractmethod
    def log_likelihood(self, W: _T) -> _T:
        """Compute log likelihood of connectivity matrices W
        
        Args:
            W: Connectivity matrices of shape [..., neurons, neurons]
        Returns:
            Log likelihoods of shape [...]
        """
        pass

class GaussianConnectivityDistribution(ConnectivityDistribution):
    
    def __init__(self, n_neurons: int, init_var: float = 1.0):
        super().__init__(n_neurons)
        
        # Mean: [n_neurons, n_neurons]
        self.mean = nn.Parameter(torch.zeros(n_neurons, n_neurons))
        
        # Covariance: [n_neurons, n_neurons, n_neurons, n_neurons]
        # Initialize as identity covariance (independent unit variances)
        cov_init = torch.zeros(n_neurons, n_neurons, n_neurons, n_neurons)
        for a in range(n_neurons):
            for b in range(n_neurons):
                cov_init[a, b, a, b] = init_var
        self.covariance = nn.Parameter(cov_init)
        
    def _flatten_mean(self) -> _T:
        """Flatten mean to [n_neurons^2]"""
        return self.mean.view(-1)
        
    def _flatten_covariance(self) -> _T:
        """Flatten covariance to [n_neurons^2, n_neurons^2]"""
        n_sq = self.n_neurons * self.n_neurons
        return self.covariance.view(n_sq, n_sq)
        
    def sample(self, batches: int, trials: int) -> _T:
        """Sample connectivity matrices of shape [batches, trials, neurons, neurons]"""
        # Flatten for multivariate normal sampling
        mean_flat = self._flatten_mean()  # [n_neurons^2]
        cov_flat = self._flatten_covariance()  # [n_neurons^2, n_neurons^2]
        
        # Add small regularization for numerical stability
        cov_reg = cov_flat + 1e-6 * torch.eye(
            self.n_neurons * self.n_neurons, 
            device=cov_flat.device
        )
        
        # Reparameterization: samples = mean + cov_reg_chol @ noise
        cov_reg_chol = torch.linalg.cholesky(cov_reg)
        noise = torch.randn(
            batches, trials, self.n_neurons * self.n_neurons,
            device=mean_flat.device, dtype=mean_flat.dtype
        )
        samples = mean_flat.unsqueeze(0).unsqueeze(0) + torch.einsum('ij,btj->bti', cov_reg_chol, noise)
        
        
        # Reshape to connectivity matrix format
        return samples.view(batches, trials, self.n_neurons, self.n_neurons)
    
    def kl_against(self, other: GaussianConnectivityDistribution) -> _T:
        """Compute KL divergence KL[self || other] between two Gaussian distributions"""
        if not isinstance(other, GaussianConnectivityDistribution):
            raise ValueError("Can only compute KL against another GaussianConnectivityDistribution")
            
        # Flatten distributions
        mu_q = self._flatten_mean()  # [n_neurons^2]
        Sigma_q = self._flatten_covariance()  # [n_neurons^2, n_neurons^2]
        mu_p = other._flatten_mean()  # [n_neurons^2]
        Sigma_p = other._flatten_covariance()  # [n_neurons^2, n_neurons^2]
        
        k = mu_q.shape[0]  # n_neurons^2
        
        # Add regularization for numerical stability
        Sigma_q_reg = Sigma_q + 1e-6 * torch.eye(k, device=Sigma_q.device)
        Sigma_p_reg = Sigma_p + 1e-6 * torch.eye(k, device=Sigma_p.device)
        
        # KL(q||p) = 0.5 * [tr(Sigma_p^-1 * Sigma_q) + (mu_p - mu_q)^T * Sigma_p^-1 * (mu_p - mu_q) - k + log(det(Sigma_p)/det(Sigma_q))]
        Sigma_p_inv = torch.inverse(Sigma_p_reg)
        
        trace_term = torch.trace(torch.mm(Sigma_p_inv, Sigma_q_reg))
        
        mean_diff = mu_p - mu_q
        quad_term = torch.dot(mean_diff, torch.mv(Sigma_p_inv, mean_diff))
        
        logdet_p = torch.logdet(Sigma_p_reg)
        logdet_q = torch.logdet(Sigma_q_reg)
        logdet_term = logdet_p - logdet_q
        
        kl = 0.5 * (trace_term + quad_term - k + logdet_term)
        
        return kl

    def log_likelihood(self, W: _T) -> _T:
        """Compute log likelihood of connectivity matrices W under multivariate Gaussian"""
        # Get the leading dimensions
        leading_shape = W.shape[:-2]
        batch_size = W[..., 0, 0].numel()  # Total number of matrices
        
        # Flatten W to [..., n_neurons^2]
        W_flat = W.view(*leading_shape, -1)
        
        # Get distribution parameters
        mean_flat = self._flatten_mean()  # [n_neurons^2]
        cov_flat = self._flatten_covariance()  # [n_neurons^2, n_neurons^2]
        
        # Add regularization
        cov_reg = cov_flat + 1e-6 * torch.eye(
            self.n_neurons * self.n_neurons, 
            device=cov_flat.device
        )
        
        # Compute log likelihood using multivariate normal formula
        k = mean_flat.shape[0]  # n_neurons^2
        
        # Flatten for batch computation
        W_batch = W_flat.view(-1, k)  # [batch_size, n_neurons^2]
        
        try:
            # Efficient batch computation
            cov_inv = torch.inverse(cov_reg)
            logdet = torch.logdet(cov_reg)
            
            # Centered data
            centered = W_batch - mean_flat.unsqueeze(0)  # [batch_size, n_neurons^2]
            
            # Quadratic form: (x - mu)^T Sigma^-1 (x - mu)
            quad_form = torch.sum(centered * torch.mm(centered, cov_inv), dim=1)  # [batch_size]
            
            # Log likelihood: -0.5 * (k*log(2π) + logdet + quad_form)
            log_probs = -0.5 * (k * math.log(2 * math.pi) + logdet + quad_form)
            
        except:
            # Fallback: assume diagonal covariance
            diag_var = torch.diagonal(cov_reg)
            centered = W_batch - mean_flat.unsqueeze(0)
            log_probs = -0.5 * torch.sum(
                centered.pow(2) / diag_var.unsqueeze(0) + torch.log(2 * math.pi * diag_var),
                dim=1
            )
        
        # Reshape back to original leading dimensions
        return log_probs.view(leading_shape)


class PointMassConnectivityDistribution(GaussianConnectivityDistribution):
    """Gaussian distribution with small epsilon ball covariance around a given mean"""
    
    def __init__(self, n_neurons: int, mean_matrix: _T, eps: float = 1e-3):
        # Initialize parent with zeros (we'll override the mean)
        super().__init__(n_neurons)
        
        # Set the mean to the provided matrix
        if mean_matrix.shape != (n_neurons, n_neurons):
            raise ValueError(f"mean_matrix must have shape ({n_neurons}, {n_neurons})")
        
        with torch.no_grad():
            self.mean.copy_(mean_matrix)
        
        self.eps = torch.tensor(eps)
        
        with torch.no_grad():
            self.covariance.zero_()
            for a in range(self.n_neurons):
                for b in range(self.n_neurons):
                    self.covariance[a, b, a, b] = self.eps
        
    def sample(self, batches: int, trials: int) -> _T:
        """Sample with current epsilon value"""
        # Simple sampling: mean + eps * randn
        mean_expanded = self.mean.unsqueeze(0).unsqueeze(0).expand(batches, trials, -1, -1)
        noise = torch.randn_like(mean_expanded) * self.eps.sqrt()
        
        return mean_expanded + noise

    def log_likelihood(self, W: _T) -> _T:
        """Compute log likelihood under point mass (small epsilon ball)"""
        # Get the leading dimensions
        leading_shape = W.shape[:-2]
        
        # Compute squared distance from mean
        diff = W - self.mean  # [..., neurons, neurons]
        squared_dist = torch.sum(diff.pow(2), dim=(-2, -1))  # [...]
        
        # Log likelihood under isotropic Gaussian with variance eps
        # log p(W) = -0.5 * [||W - μ||²/ε + k*log(2π ε)]
        k = self.n_neurons * self.n_neurons
        
        log_probs = -0.5 * (
            squared_dist / self.eps +                       # ||W - μ||²/ε
            k * torch.log(2 * math.pi * self.eps)          # k*log(2π ε)
        )
        
        return log_probs  # [...]

class DiagonalGaussianConnectivityDistribution(GaussianConnectivityDistribution):
    """Gaussian distribution with diagonal covariance (only variances, no covariances)"""
    
    def __init__(self, n_neurons: int, init_var: float = 1.0):
        # Initialize parent without calling super().__init__() to avoid full covariance initialization
        ConnectivityDistribution.__init__(self, n_neurons)
        
        # Mean: [n_neurons, n_neurons]
        self.mean = nn.Parameter(torch.zeros(n_neurons, n_neurons))
        
        # Only store diagonal variances: [n_neurons, n_neurons]
        # Each element represents the variance of the corresponding W[i,j]
        self.log_variances = nn.Parameter(torch.log(torch.full((n_neurons, n_neurons), init_var)))
        
    @property
    def variances(self) -> _T:
        """Get current variances (always positive via exp)"""
        return torch.exp(self.log_variances)
    
    def _flatten_covariance(self) -> _T:
        """Create diagonal covariance matrix [n_neurons^2, n_neurons^2]"""
        variances_flat = self.variances.view(-1)  # [n_neurons^2]
        return torch.diag(variances_flat)  # [n_neurons^2, n_neurons^2]
    
    def sample(self, batches: int, trials: int) -> _T:
        """Sample connectivity matrices using reparameterization trick"""
        # Get mean and standard deviations
        mean_flat = self._flatten_mean()  # [n_neurons^2]
        std_flat = self.variances.view(-1).sqrt()  # [n_neurons^2]
        
        # Reparameterization trick: sample = mean + std * noise
        noise = torch.randn(
            batches, trials, self.n_neurons * self.n_neurons,
            device=mean_flat.device, dtype=mean_flat.dtype
        )
        
        samples = mean_flat.unsqueeze(0).unsqueeze(0) + std_flat.unsqueeze(0).unsqueeze(0) * noise
        
        # Reshape to connectivity matrix format
        return samples.view(batches, trials, self.n_neurons, self.n_neurons)
    
    def kl_against(self, other: GaussianConnectivityDistribution) -> _T:
        """Compute KL divergence KL[self || other]"""
        if isinstance(other, DiagonalGaussianConnectivityDistribution):
            # Both are diagonal - use efficient computation
            return self._kl_against_diagonal(other)
        else:
            # Other has full covariance - fall back to parent method
            return super().kl_against(other)
    
    def _kl_against_diagonal(self, other: 'DiagonalGaussianConnectivityDistribution') -> _T:
        """Efficient KL computation when both distributions are diagonal"""
        # Flatten means and variances
        mu_q = self._flatten_mean()  # [n_neurons^2]
        var_q = self.variances.view(-1)  # [n_neurons^2]
        mu_p = other._flatten_mean()  # [n_neurons^2]
        var_p = other.variances.view(-1)  # [n_neurons^2]
        
        # For diagonal Gaussians: KL(q||p) = 0.5 * sum[var_q/var_p + (mu_p-mu_q)^2/var_p - 1 + log(var_p/var_q)]
        kl = 0.5 * torch.sum(
            var_q / var_p +                           # variance ratio
            (mu_p - mu_q).pow(2) / var_p +           # squared mean difference scaled by prior variance
            torch.log(var_p / var_q) -               # log variance ratio
            1.0                                       # constant term
        )
        
        return kl

    def log_likelihood(self, W: _T) -> _T:
        """Compute log likelihood under diagonal Gaussian (more efficient)"""
        # Get the leading dimensions
        leading_shape = W.shape[:-2]
        
        # Flatten W to [..., n_neurons^2]
        W_flat = W.view(*leading_shape, -1)
        
        # Get distribution parameters
        mean_flat = self._flatten_mean()  # [n_neurons^2]
        var_flat = self.variances.view(-1)  # [n_neurons^2]
        
        # Compute log likelihood element-wise (since covariance is diagonal)
        # log p(W) = -0.5 * sum[(W - μ)²/σ² + log(2π σ²)]
        centered = W_flat - mean_flat  # [..., n_neurons^2]
        
        log_probs = -0.5 * torch.sum(
            centered.pow(2) / var_flat +                    # (W - μ)²/σ²
            torch.log(2 * math.pi * var_flat),             # log(2π σ²)
            dim=-1  # Sum over n_neurons^2 dimension
        )
        
        return log_probs  # [...]
