import torch
from torch import nn
from torch import Tensor as _T
from torch.distributions import MultivariateNormal

from math import sqrt


def kl_between_constrained_gaussians(mu1: _T, mu2: _T, Sigma1_diag: _T, Sigma2_chol: _T) -> _T:
    """
    mu1, mu2 are [..., dim]
    Sigma1_diag is a [..., dim], giving just the diagonal
    Sigma2_chol is a cholesky [..., dim, dim]
    output of shape [...]

    We don't need the logdet_Sigma1 term for our purposes, but we keep it for debugging of negative KL
    """

    assert mu1.shape == mu2.shape
    dims = mu1.shape[-1]
    assert list(Sigma1_diag.shape) == list(mu2.shape)
    assert list(Sigma2_chol.shape) == list(mu2.shape) + [dims]
    assert (torch.triu(Sigma2_chol, 1) == 0.0).all()

    logdet_Sigma1 = Sigma1_diag.log().sum(-1)
    logdet_Sigma2 = 2 * Sigma2_chol.diagonal(0, -1, -2).log().sum(-1)
    det_term = logdet_Sigma2 - logdet_Sigma1

    inv_Sigma2_chol: _T = torch.linalg.inv(Sigma2_chol)
    inv_Sigma2 = inv_Sigma2_chol.transpose(-1, -2) @ inv_Sigma2_chol
    # inv_Sigma2_alt = torch.linalg.inv(Sigma2_chol @ Sigma2_chol.transpose(-1, -2))

    trace_term = (inv_Sigma2.diagonal(dim1 = -1, dim2 = -2) * Sigma1_diag).sum(-1)

    mean_diff = (mu2 - mu1)
    mean_diff_scaled = (inv_Sigma2_chol.transpose(-1, -2) @ mean_diff.unsqueeze(-1))
    inner_prod_term = (mean_diff_scaled).square().squeeze(-1).sum(-1)

    kl = 0.5 * (det_term - dims + trace_term + inner_prod_term)
    
    if (kl < 0.0).any():
        import pdb; pdb.set_trace()
        raise Exception('Invalid KL! Contains negative values')
    if kl.isnan().any():
        import pdb; pdb.set_trace()
        raise Exception('Invalid KL! Contains nan values')

    return kl


def cross_entropy_of_gaussians(mu_p: _T, Sigma_p: _T, y_samples: _T):
    """
    Used in reconstruction term:
        - < < log p(y | x) >_{ q1(x | y) } >_{qB(y | Z)}

    y_samples of shape [..., dim]
    mu_p, Sigma_p of shapes [..., dim] and [..., dim, dim], with 1-1 correspondance
        i.e. p(y_samples[...,:] | x) = N(mu_p[...,:], Sigma_p[...,:,:])

    output of shape [...]
    """
    return - MultivariateNormal(loc = mu_p, covariance_matrix=Sigma_p).log_prob(y_samples)


def run_forward_process(y_samples: _T, sigma2x_schedule: _T, sigma2x_orthogonal_multiplier: float, linking_matrix: _T, projection_basis: _T):
    """
    Apply x_0 = linking_matrix @ y deterministically, then roll out samples of q_t(x_t | x_{t-1}) repeatedly until we sample x_T | x_{T-1}

    Apply z_0 = projection_basis @ x_0
        First two dimensions of z_0 are noised with variance sigma2x_schedule
        Remaining dimensions of z_0 are noised with variance sigma2x_schedule * sigma2x_orthogonal_multiplier
    
    Noise all then transform back with projection_basis.T @ x_0

    y_samples of shape [..., dim_y]
    sigma2x_schedule of shape [T] and sigma2_t = sigma2x_schedule[t-1]
        i.e. q(x_t | x_{t-1}) = N( x_t; (1 - sigma2_t) * x_{t-1}, sigma2_t * I )
        e.g. q(x_2 | x_1) uses sigma2x_schedule[1] ... q(x_T | x_{T - 1}) uses sigma2x_schedule[T-1]

    Output of shape [..., T, dim_x]
    """
    x_0 = y_samples @ linking_matrix    # [..., dim_x]
    z_0 = x_0 @ projection_basis.T
    # assert torch.isclose(z_0[:,:2], y_samples).all()

    std_schedule_vector = sigma2x_schedule.unsqueeze(-1).repeat(1, x_0.shape[-1])           # [T, dim_x]
    std_schedule_vector[:,2:] = (std_schedule_vector[:,2:] * sigma2x_orthogonal_multiplier).sqrt()
    std_schedule_vector = std_schedule_vector[*[None for _ in x_0.shape[:-1]]]   # [1s like ..., T, dim_x]

    all_z_samples = [
        (torch.sqrt(1. - std_schedule_vector[...,0,:]) * z_0) + (std_schedule_vector[...,0,:] * torch.randn_like(z_0)) # x1, the first actual latent
    ]
    for t in range(1, std_schedule_vector.shape[-2]):
        relevant_std = std_schedule_vector[...,t,:]
        all_z_samples.append(
            (torch.sqrt(1 - relevant_std ** 2) * all_z_samples[-1]) + (relevant_std * torch.randn_like(all_z_samples[-1]))
        )
    
    all_z_samples = torch.stack(all_z_samples, -2)
    all_x_samples = all_z_samples @ projection_basis
    assert all_x_samples.shape == all_z_samples.shape

    return all_x_samples



def generate_reversed_noising_posteriors(y_samples: _T, x_samples: _T, sigma2x_schedule: _T, sigma2x_orthogonal_multiplier: float, linking_matrix: _T, projection_basis: _T):
    """
    NOTE: the below is old documentation that I can't be bothered changing.
    
    Now, we are doing almost exactly the same but in z space (just rotated x space), see run_forward_process for more details.

    barsigma2s is now shape [..., T-1, dim_x] again giving the diagonal, because noising is anisotropic
        NB: dim_z == dim_x

    XXX XXXXXXXXXXXXXXXXXXXX XXX

    y_samples of shape [..., dim_y]
    x_samples of shape [..., T, dim_x], output of run_forward_process in forward order, i.e. x_samples[...,0,:] = x1 and last one (x_samples[...,-1,:] = xT) is "ignored"
    sigma2x_schedule of shape [T], same as as input to run_forward_process
        i.e. sigma2x_schedule[0] was used to generate x_samples[...,0,:], etc...

    In the below we define x_0 = linking_matrix @ y (deterministically)

    For the KL terms we need q(x_{t-1} | x_t, y)
    This is N(x_{t-1}; barmu_t, barsigma2_t I), which are functions of all the sigma2xs leading up to that timestep
        Remember that sigma2_1 = sigma2x_schedule[0] is used to noise x_0 to x_1, the first x sample in the forward process!

    We only need q(x_1 | x_2, y) up to q(x_{T-1} | x_T, y), where dependence on y effectively becomes dependence on x_0!
    
    Outputs:
        barmus: [..., T-1, dim_x]
        barsigma2s: [..., T-1] giving the diagonal
        
        Here, quantity[...,t,:,:] is for q(x_{t+1} | x_{t+2}, y)
        e.g. quantity[...,0,:,:] is for q(x_1 | x_2, y) ... quantity[...,-1,:,:] = quantity[...,T-2,:,:] is for q(x_{T-1} | x_{T-2}, y)

        Consult cross_entropy_terms_on_reverse_process docstring for more info!
    """

    ellipsis_dims = list(y_samples.shape[:-1])

    x_dim = linking_matrix.shape[1]
    sigma2x_schedule_vector = sigma2x_schedule.unsqueeze(-1).repeat(1, x_dim)
    sigma2x_schedule_vector[:,2:] = sigma2x_schedule_vector[:,2:] * sigma2x_orthogonal_multiplier   # [T, x_dim]
    sigma2x_schedule_vector = sigma2x_schedule_vector[*[None]*len(ellipsis_dims)]                   # [1,...,1, T, x_dim]

    a_t = torch.cumprod(torch.sqrt(1. - sigma2x_schedule_vector), -2)   # [1...1, T, x_dim]
    b_t = 1. - torch.cumprod((1. - sigma2x_schedule_vector), -2)        # [1...1, T, x_dim]

    x_0 = (y_samples @ linking_matrix).unsqueeze(-2)    # [...,1,dim_x]
    z_0 = x_0 @ projection_basis.T
    # assert torch.isclose(z_0[:,0,:2], y_samples).all()

    a_t_minus_1 = a_t[...,:-1,:]            # all [1,...,1,T-1,x_dim] -- recall: for q(x_{t-1} | x_t, y), t = 2,...,T
    b_t_minus_1 = b_t[...,:-1,:]
    b_t_current = b_t[...,1:,:]

    x_t = x_samples[...,1:,:]               # [..., T-1, x_dim]
    z_t = x_t @ projection_basis.T
    
    z_barsigma2s_flat = b_t_minus_1 * sigma2x_schedule_vector[...,1:,:] / b_t_current       # [..., T-1,x_dim]

    # [...,T-1,dim_x]
    z_barmus = z_barsigma2s_flat * (
        (a_t_minus_1 / b_t_minus_1 * z_0) + 
        ((1. - sigma2x_schedule_vector[...,1:,:]).sqrt() / sigma2x_schedule_vector[...,1:,:] * z_t)
    )
    barsigma2s = z_barsigma2s_flat.repeat(*ellipsis_dims, 1, 1)

    return z_barmus, barsigma2s



def cross_entropy_terms_on_reverse_process(
    y_samples: _T,
    x_mu_ps: _T, x_Sigma_chol_ps: _T, y_mu_p: _T, y_Sigma_p: _T,
    q_reversed_z_mu_ts: _T, q_reversed_z_Sigma_ts: _T,
    projection_basis: _T
):
    """
    Inputs from:
        Outputs of BNS sampling:
            y_samples of shape [..., dim_y], same as was input to run_forward_process

        (Reversed) outputs of denoising process (RNN)
            x_mu_ps of shape [..., T-1, dim_x]
            x_Sigma_chol_ps of shape [..., T-1, dim_x, dim_x]
                both in forward order -- opposite of their actual generation!
                i.e. p(x_t | x_{t+1}) = N(x_mu_ps[t-1], x_Sigma_chol_ps[t-1])
                    e.g. p(x_1 | x_2) = N(x_mu_ps[0], x_Sigma_chol_ps[0])
                    and  p(x_{T-1} | x_T) = N(x_mu_ps[-1], x_Sigma_chol_ps[-1])

            y_mu_p = [..., dim_y]
            y_Sigma_p = [..., dim_y, dim_y]

            projection_basis will be used to rotate them into z space!

        Outputs of generate_reversed_noising_posteriors:
            q_reversed_z_mu_ts of shape [..., T-1, dim_x]
            q_reversed_z_Sigma_ts of shape [..., T-1, dim_x, dim_x] - full diagonal since noising in z space is now anisotropic
                both in forward order as well, 
                i.e. q(x_t | x_{t+1}, y) = N(q_reversed_mu_ts[t-1], q_reversed_Sigma_ts[t-1])
                    e.g. q(x_1 | x_2, y) = N(q_reversed_mu_ts[0], q_reversed_Sigma_ts[0])
                    e.g. q(x_{T-1} | x_T, y) = N(q_reversed_mu_ts[-1], q_reversed_Sigma_ts[-1])

    Outputs of shapes:
        reconstruction_cross_entropy [...]
        all_kl_terms [..., T-1]
    """
    # XXX: can cut down on all of this if x_Sigma_ps is spherical!
    z_mu_ps = x_mu_ps @ projection_basis.T
    x_Sigma_ps = torch.einsum('btik,btkj->btij', x_Sigma_chol_ps, x_Sigma_chol_ps)
    z_Sigma_ps = torch.einsum('btik,kj->btij', torch.einsum('ik,btkj->btij', projection_basis, x_Sigma_ps), projection_basis.T)
    z_Sigma_chol_ps = torch.linalg.cholesky(z_Sigma_ps)

    reconstruction_cross_entropy = cross_entropy_of_gaussians(y_mu_p, y_Sigma_p, y_samples)
    kl_terms = kl_between_constrained_gaussians(
        mu1=q_reversed_z_mu_ts,
        mu2=z_mu_ps,
        Sigma1_diag=q_reversed_z_Sigma_ts,
        Sigma2_chol=z_Sigma_chol_ps
    )
    return {
        "reconstruction_cross_entropy": reconstruction_cross_entropy, 
        "all_kl_terms": kl_terms
    }



