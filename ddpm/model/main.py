from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Dict, Optional, List

from abc import ABC, abstractmethod

from ddpm.model.time_repr import TimeEmbeddingBlock
from ddpm.model.residual import VectoralResidualModel, UNetResidualModel
from ddpm.model.input import InputModelBlock





class DDPMReverseProcessBase(nn.Module, ABC):
    """
    Time varying sigma schedule:
        sigma2xt_schedule is beta_t in the original literature, of shape [T]
        sigma2xt_schedule[i] gives \\sigma^2_{i+1} e.g. sigma2xt_schedule[0] gives \\sigma^2_1 etc.
    """

    def __init__(
        self,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel | UNetResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        target_variance: float,
        device = 'cuda'
    ) -> None:
        super().__init__()

        self.sample_shape = sample_shape

        self.T = len(sigma2xt_schedule)

        self.residual_model = residual_model
        # self.time_embeddings = nn.Embedding(self.T, time_embedding_size)
        self.time_embeddings = TimeEmbeddingBlock(self.T, time_embedding_size, device=device)
        self.input_model = input_model
        assert input_model.network_input_size == residual_model.input_size,\
            f"Got an input model (sensory -> residual network) with output size {input_model.network_input_size}, but a residual model that accepts inputs of size {residual_model.input_size}"

        # Incremental noising
        self.target_variance = target_variance
        self.sigma2xt_schedule = sigma2xt_schedule          # t = 1, ..., T
        assert len(sigma2xt_schedule.shape) == 1
        self.t_schedule = torch.arange(self.T)   # Indexes self.time_embeddings.time_embs
        self.incremental_modulation_schedule = torch.sqrt(1.0 - self.sigma2xt_schedule)
        self.std_schedule = torch.sqrt(self.sigma2xt_schedule)

        # One-shot noising
        self.a_t_schedule = torch.cumprod(self.incremental_modulation_schedule, 0)  # t = 1, ..., T
        self.root_b_t_schedule = torch.sqrt(1.0 - torch.square(self.a_t_schedule))      # t = 1, ..., T

        # Generation
        self.base_std = self.root_b_t_schedule[-1]  # scalar, final variance = b_T
        noise_scaler_schedule = self.root_b_t_schedule[:-1] / self.root_b_t_schedule[1:] * self.std_schedule[1:]    # t = 2, ..., T
        self.noise_scaler_schedule = torch.concat([torch.zeros(1, device = noise_scaler_schedule.device), noise_scaler_schedule], 0)    # Still t = 2, ..., T but with a zero appended to the front for ease!
        self.base_samples_scaler_schedule = 1.0 / self.incremental_modulation_schedule                  # t = 1, ..., T
        self.residual_scaler_schedule = self.sigma2xt_schedule / self.incremental_modulation_schedule / self.root_b_t_schedule
        
        ## Training
        #mse_scaler_schedule_num = (1 - self.a_t_schedule[1:] ** 2.0) * self.sigma2xt_schedule[1:]    # t = 2, ..., T
        #mse_scaler_schedule_denom = 2 * (1 - self.a_t_schedule[:-1] ** 2.0) * (1.0 - self.sigma2xt_schedule[1:])    # t = 2, ..., T
        #self.mse_scaler_schedule = mse_scaler_schedule_num / mse_scaler_schedule_denom

        ## Convinience                
        self.reshaped_a_t_schedule = self.a_t_schedule[:, *[None for _ in self.sample_shape]] # [T, <1 for each of shape x>]
        self.reshaped_root_b_t_schedule = self.root_b_t_schedule[:, *[None for _ in self.sample_shape]] # [T, <1 for each of shape x>]


    def to(self, *args, **kwargs):
        self.sigma2xt_schedule = self.sigma2xt_schedule.to(*args, **kwargs)
        self.t_schedule = self.t_schedule.to(*args, **kwargs)
        self.incremental_modulation_schedule = self.incremental_modulation_schedule.to(*args, **kwargs)
        self.a_t_schedule = self.a_t_schedule.to(*args, **kwargs)
        self.root_b_t_schedule = self.root_b_t_schedule.to(*args, **kwargs)
        # self.mse_scaler_schedule = self.mse_scaler_schedule.to(*args, **kwargs)
        self.noise_scaler_schedule = self.noise_scaler_schedule.to(*args, **kwargs)
        return super(DDPMReverseProcessBase, self).to(*args, **kwargs)


    @abstractmethod
    def noise(self, x_0: _T) -> Dict[str, _T]:
        """
        x_0 of shape [..., <shape x>]

        Requires output contains at least x_t and epsilon, with standard definitions given in OneShotDDPMReverseProcess.noise
        """
        raise NotImplementedError
    
    @abstractmethod
    def residual(self, x_samples: _T, network_input: _T, **kwargs_for_residual_model) -> Dict[str, _T]:  # epsilon: Optional[_T] = None, 
        """
        Given a full trajectory of x_t, predict the *one-shot* epsilons that produced them
        In the teacher-forcing case, we are actually predicting the *effective* epsilons that would have cause such a one-shot noising pattern
        """
        raise NotImplementedError
    
    def denoise_one_step(self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float):
        """
        t_idx indexes time backwards, so ranges from 1 to T

        x_t_plus_1 is samples denoised so far - of shape [..., 1, <shape x>]
        predicted_residual is epsilon hat - should be the same size

        HOWEVER, we do not check the shapes, because some child classes, e.g. LinearSubspaceTeacherForcedDDPMReverseProcess,
            apply this iterative processing to some shared space of samples and auxiliary computation variables...

        TODO: double computation with early_x0_pred!
        """
        assert t_idx > 0 and t_idx <= self.T

        # assert list(x_t_plus_1.shape) == [num_samples, 1, *self.sample_shape] == list(predicted_residual.shape)
        noise = noise_scaler * torch.randn_like(x_t_plus_1)
        scaled_noise = noise * self.noise_scaler_schedule[-t_idx]
        scaled_base_samples = x_t_plus_1 * self.base_samples_scaler_schedule[-t_idx]
        scaled_residual = self.residual_scaler_schedule[-t_idx] * predicted_residual # [..., 1, dim x]

        one_step_denoise = scaled_base_samples - scaled_residual + scaled_noise
        early_x0_pred = (x_t_plus_1 - self.root_b_t_schedule[-t_idx] * predicted_residual) / self.a_t_schedule[-t_idx]

        return one_step_denoise, early_x0_pred

    @abstractmethod
    def generate_samples(self, network_input: _T, samples_shape: Optional[List[int]] = None, base_samples: Optional[_T] = None, noise_scaler: float = 1.0) -> Dict[str, _T]:
        """
        input_vector of shape [..., <shape Z>]
        If provided, base_samples of shape [..., <shape x>]
        Otherwise, give ... = samples_shape

        sample_trajectory of shape [..., T, <shape x>]
        samples of shape [..., <shape x>]
        early_x0_preds of shape [..., T, <shape x>]
        """
        raise NotImplementedError



class OneShotDDPMReverseProcess(DDPMReverseProcessBase):
    """
    Standard DDPM noising - whole space noised together, space noised one-shot

    At training time, residuals are predicted for each timestep independently
    """

    def noise(self, x_0: _T) -> Dict[str, _T]:
        """
        x_0 of shape [..., <shape x>]

        Both outputs of shape [..., T, dim x]
            x_t = one-shot noising
            epsilon = actual epsilon used for one-shot noising
        """
        assert list(x_0.shape[-len(self.sample_shape):]) == self.sample_shape, f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        num_extra_dim = len(x_0.shape) - len(self.sample_shape)
        x_0 = x_0.unsqueeze(num_extra_dim).repeat(*[1]*num_extra_dim, self.T, *[1]*len(self.sample_shape))
        epsilon = torch.randn_like(x_0)

        # [..., T, dim x]
        x_t_means = self.reshaped_a_t_schedule[*[None]*num_extra_dim] * x_0 
        x_t = x_t_means + self.reshaped_root_b_t_schedule[*[None]*num_extra_dim] * epsilon

        return {
            'x_t': x_t,
            'epsilon': epsilon
        }

    def residual(self, x_samples: _T, network_input: _T, kwargs_for_residual_model = {}) -> Dict[str, _T]:
        """
        x_samples of shape [..., T, <shape x>]
        network_input of shape [..., <shape Z>]       --> same for all timesteps in this class!
        """
        num_extra_dim = len(x_samples.shape) - len(self.sample_shape) - 1 # including time now
        batch_shape = x_samples.shape[:num_extra_dim]
        input_vector: _T = self.input_model(network_input)
        
        assert tuple(input_vector.shape) == (*batch_shape, self.residual_model.input_size,), f"Expected input_vector shape to be {(*batch_shape, self.residual_model.input_size)} but got {tuple(input_vector.shape)}"

        assert tuple(x_samples.shape) == (*batch_shape, self.T, *self.sample_shape),\
            f"Expected x_samples shape to end in {self.sample_shape} but got {x_samples.shape}"

        reshaped_input_vector = input_vector.unsqueeze(-2).repeat(*[1 for _ in batch_shape], self.T, 1)
        t_embeddings = self.time_embeddings(self.t_schedule)
        epsilon_hat = self.residual_model(x = x_samples, t_embeddings_schedule = t_embeddings, input_vector = reshaped_input_vector, **kwargs_for_residual_model)    # [..., T, <sample shape>]
        assert x_samples.shape == epsilon_hat.shape

        return {'epsilon_hat': epsilon_hat}

    @torch.no_grad()
    def generate_samples(self, network_input: _T, samples_shape: Optional[List[int]] = None, base_samples: Optional[_T] = None, noise_scaler: float = 1.0) -> Dict[str, _T]:
        """
        input_vector of shape [..., <shape Z>]
        If provided, base_samples of shape [..., <shape x>]
        Otherwise, give ... = samples_shape

        sample_trajectory of shape [..., T, <shape x>]
        samples of shape [..., <shape x>]
        early_x0_preds of shape [..., T, <shape x>]
        """
        
        assert (samples_shape is None) != (base_samples is None)

        if base_samples is None:
            base_samples = torch.randn(*samples_shape, *self.sample_shape, device = self.sigma2xt_schedule.device) * self.base_std
        else:
            samples_shape = base_samples.shape[:-len(self.sample_shape)]
            assert tuple(base_samples.shape) == (*samples_shape, *self.sample_shape)
        
        input_vector = self.input_model(network_input)
        assert tuple(input_vector.shape) == (*samples_shape, self.residual_model.input_size,), \
            f"Expected input_vector shape to be just {(self.residual_model.input_size)} but got {tuple(input_vector.shape)}"
            
        input_vector = input_vector.unsqueeze(-2)         # [..., 1, Di]
        
        base_samples = base_samples.unsqueeze(len(samples_shape))                # [..., 1, D]
        t_embeddings = self.time_embeddings(self.t_schedule)

        sample_trajectory = []
        early_x0_preds = []

        for t_idx in range(1, self.T + 1):

            t_embedding = t_embeddings[-t_idx][None]            # [1, Dt]

            predicted_residual = self.residual_model(base_samples, t_embedding, input_vector)

            base_samples, early_x0_pred = self.denoise_one_step(t_idx, base_samples, predicted_residual, noise_scaler)

            sample_trajectory.append(base_samples.detach().cpu())
            early_x0_preds.append(early_x0_pred.detach().cpu())
        
        sample_trajectory = torch.concat(sample_trajectory, len(samples_shape)) # [..., T, <shape x>]
        early_x0_preds = torch.concat(early_x0_preds, len(samples_shape)) # [..., T, <shape x>]

        return {
            'sample_trajectory': sample_trajectory,
            'samples': base_samples.squeeze(-2),
            'early_x0_preds': early_x0_preds,
        }
            


class TeacherForcedDDPMReverseProcessBase(DDPMReverseProcessBase):
    """
    Now, we are doing sequential noising, which will be denoised alongside some
    dynamical computations in some auxiliary space

    The auxiliary computation is not trained directly, but only via the denoising
    of the actual variable of interest. As before, this equates to estimate some residual epsilon

    However this time, because noising is sequential and not one-shot, we are estimating
    a *effective* epsilon

    How the denoising and the auxiliary computations are combined are manifold, but 
    the sequential denoising is shared, so we put it here!
    """

    def __init__(self, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, device='cuda') -> None:
        super().__init__(sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, device)

    @torch.no_grad()
    def noise(self, x_0: _T) -> Dict[str, _T]:
        """
        x_0 of shape [..., <shape x>]

        All outputs of shape [..., T, dim x]
            x_t = noising trajectory (actually a trajectory here)
            epsilon_actual = actual incrementental epsilons added to samples (before scaling and modulation)
            epsilon = effective epsilon, which would have taken us from x_0 to x_t one shot (after scaling and modulation)
        """
        assert list(x_0.shape[-len(self.sample_shape):]) == self.sample_shape, f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        epsilon_actual = torch.randn(self.T, *x_0.shape).to(x_0.device)
        num_extra_dim = len(x_0.shape) - len(self.sample_shape)
        
        # Construct trajectory
        trajectory = [x_0]
        for t, eps in enumerate(epsilon_actual):    # Forward time
            trajectory.append((trajectory[-1] * self.incremental_modulation_schedule[t]) + (self.std_schedule[t] * eps))
        x_t = torch.stack(trajectory[1:], num_extra_dim)

        # Effective epsilons, which should be predicted
        epsilon_effective = (x_t - (self.reshaped_a_t_schedule * x_0.unsqueeze(num_extra_dim))) / self.reshaped_root_b_t_schedule

        return {
            'x_t': x_t,
            'epsilon_actual': epsilon_actual,
            'epsilon': epsilon_effective,
        }



class LinearSubspaceTeacherForcedDDPMReverseProcess(TeacherForcedDDPMReverseProcessBase):
    """
    A special case where the sample space is embedded as a linear subspace, with the remaining directions
    used as the auxiliary computation variables

    Generates its own buffer auxiliary_embedding_matrix, which should not be confused with VectoralEmbeddedExampleSampleGenerator.linking_matrix!
    """

    def __init__(self, seperate_output_neurons: bool, stabilise_nullspace: bool, sample_ambient_dim: int, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, device='cuda') -> None:
        super().__init__(sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, device)

        self.do_teacher_forcing = True

        assert len(sample_shape) == 1 and (sample_ambient_dim >= sample_shape[0])
        assert isinstance(self.residual_model, VectoralResidualModel) and (self.residual_model.state_space_size == sample_ambient_dim)

        self.sample_ambient_dim = sample_ambient_dim

        self.stabilise_nullspace = stabilise_nullspace
        if stabilise_nullspace:
            self.euler_alpha = 0.1

        if sample_ambient_dim > sample_shape[0]:
            if seperate_output_neurons:
                orth = torch.eye(sample_ambient_dim)
            else:
                gaus = torch.randn(sample_ambient_dim, sample_ambient_dim)
                svd = torch.linalg.svd(gaus)
                orth = svd[0] @ svd[2]
        else:
            orth = torch.eye(sample_shape[0])
        self.register_buffer('auxiliary_embedding_matrix', orth[:sample_shape[0]])  # [little space, big space]
        self.register_buffer('sample_subspace_accessor', orth[:sample_shape[0]].T @ orth[:sample_shape[0]])  # [big space, big space]
        self.register_buffer('behaviour_nullspace', orth[sample_shape[0]:])             # [extra dims, big space]
        self.register_buffer('behaviour_nullspace_accessor', orth[sample_shape[0]:].T @ orth[sample_shape[0]:]) # [big space, big space]

    def to(self, *args, **kwargs):
        self.sample_subspace_accessor = self.sample_subspace_accessor.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def denoise_one_step(self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float):
        if self.stabilise_nullspace:
            coeff = (1 - self.euler_alpha - self.base_samples_scaler_schedule[-t_idx]) / self.residual_scaler_schedule[-t_idx]
            stabilising_correction = coeff * (x_t_plus_1 @ self.behaviour_nullspace_accessor)
            predicted_residual = predicted_residual - stabilising_correction
        return super().denoise_one_step(t_idx, x_t_plus_1, predicted_residual, noise_scaler)

    def residual(self, x_samples: _T, network_input: _T, initial_state: Optional[_T] = None, kwargs_for_residual_model = {}) -> Dict[str, _T]:
        """
        x_samples of shape [..., T, <shape x>]
            Importantly, these now define continuous, gradually noised sample trajectories, not one-shot noising from the GT samples
        
        network_input of shape [..., <shape Z>]       
            still the same for all timesteps, if the network is to be used for computation beforehand, it can should do so 
            before calling this method, i.e. encoded in initial_state - see below

        initial_state of shape [..., ambient space dim]
            starting combined state of sample variable and auxiliary computation variables, all embedded into the larger ambient space

        Key differences here compared to OneShotDDPMReverseProcess:
            - Embedding: the x_samples are embedded into the ambient space before passing it throught the residual model prediction, 
                then extracted to give epsilon_hat, as we only train on that linear subspace
            - Teacher-forcing: epsilon_hat prediction is unfortunately no longer parallelised across timesteps, but is instead done autoregressively.
                Furthermore, at each step, the sample subspace of the ambient space is subtracted, and instead the real x_samples value is added in its place.
        """
        num_extra_dim = len(x_samples.shape) - len(self.sample_shape) - 1 # including time now
        batch_shape = x_samples.shape[:num_extra_dim]
        input_vector: _T = self.input_model(network_input)

        input_vector = input_vector.unsqueeze(-2)

        assert tuple(input_vector.shape) == (*batch_shape, 1, self.residual_model.input_size,), f"Expected input_vector shape to be {(*batch_shape, self.residual_model.input_size)} but got {tuple(input_vector.shape)}"
        assert tuple(x_samples.shape) == (*batch_shape, self.T, *self.sample_shape), f"Expected x_samples shape to end with {self.sample_shape} but got {x_samples.shape}"

        t_embeddings = self.time_embeddings(self.t_schedule)
        embedded_samples = x_samples @ self.auxiliary_embedding_matrix        # [..., T, ambient space dim] --> will be used for teacher forcing

        if initial_state is None:
            initial_state = torch.randn(*batch_shape, self.sample_ambient_dim, device = self.sigma2xt_schedule.device) * self.base_std
        else:
            assert tuple(initial_state.shape) == (*batch_shape, self.sample_ambient_dim), f"Expected initial_state shape to end with {self.sample_ambient_dim} but got {tuple(initial_state.shape)}"

        one_step_denoising = initial_state.unsqueeze(-2) # [..., 1, ambient space dim]
        all_predicted_residuals = []
        all_subspace_trajectories = []

        for t_idx in range(1, self.T + 1):

            # If required, correct the state. This is equivalent to just replacing the embedded_predicted_residual
            if self.do_teacher_forcing:
                # Course correct in the linear sample subspace ---> (sample_removed_one_step_denoising @ self.auxiliary_embedding_matrix.T).abs().max() is very small
                sample_removed_one_step_denoising = one_step_denoising - (one_step_denoising @ self.sample_subspace_accessor)
                # NB: this indexing will have to be fixed for structed data...
                one_step_denoising = sample_removed_one_step_denoising + embedded_samples[..., [-t_idx], :]  # ((one_step_denoising @ self.auxiliary_embedding_matrix.T) - x_samples[:, :, [-t_idx], ...]).abs().max() is very small

            # Denoise in the full ambient space for one step: one_step_denoising, early_embedded_x0_pred both of shape [..., 1, ambient space dim]
            t_embedding = t_embeddings[-t_idx][None]
            embedded_predicted_residual = self.residual_model(one_step_denoising, t_embedding, input_vector)
            one_step_denoising, early_embedded_x0_pred = self.denoise_one_step(t_idx, one_step_denoising, embedded_predicted_residual, noise_scaler=1.0)
            # early_x0_pred = early_embedded_x0_pred @ self.auxiliary_embedding_matrix.T      # [..., 1, sample dim]

            # Only these directions have the actual interpretation of a 'predicted residual'
            predicted_residual = embedded_predicted_residual @ self.auxiliary_embedding_matrix.T
            all_predicted_residuals.append(predicted_residual)

            subspace_activity = one_step_denoising @ self.auxiliary_embedding_matrix.T
            all_subspace_trajectories.append(subspace_activity)


        epsilon_hat = torch.concat(all_predicted_residuals[::-1], num_extra_dim)    # forward (diffusion) time!
        assert x_samples.shape == epsilon_hat.shape
        
        subspace_trajectories = torch.concat(all_subspace_trajectories, num_extra_dim)    # keep as reverse (denoising) time!
        assert x_samples.shape == subspace_trajectories.shape

        return {'epsilon_hat': epsilon_hat, 'subspace_trajectories': subspace_trajectories}

    @torch.no_grad()
    def generate_samples(self, network_input: _T, samples_shape: Optional[List[int]] = None, base_samples: Optional[_T] = None, noise_scaler: float = 1.0, kwargs_for_residual_model = {}) -> Dict[str, _T]:
        """
        Only difference to OneShotDDPMReverseProcess.generate_samples is that the denoising is done in the ambient space, not in the
            sample space. Samples are decoded at the end

        input_vector of shape [..., <shape Z>] purely out of convinience for us!
        If provided, base_samples of shape [..., ambient space dim] !!!
        Otherwise, give B = num_samples

        sample_trajectory of shape [..., T, <shape x>]
        samples of shape [..., <shape x>]
        early_x0_preds of shape [..., T, <shape x>]
        """
        
        assert (samples_shape is None) != (base_samples is None)

        if base_samples is None:
            base_samples = torch.randn(*samples_shape, self.sample_ambient_dim, device = self.sigma2xt_schedule.device) * self.base_std
        else:
            samples_shape = base_samples.shape[:-1]
            assert tuple(base_samples.shape) == (*samples_shape, self.sample_ambient_dim)

        input_vector = self.input_model(network_input)
        assert tuple(input_vector.shape) == (*samples_shape, self.residual_model.input_size,),\
            f"Expected input_vector shape to be just {(self.residual_model.input_size)} but got {tuple(input_vector.shape)}"
        
        input_vector = input_vector.unsqueeze(-2)         # [..., 1, Di]
        
        base_samples = base_samples.unsqueeze(len(samples_shape))   # [..., 1, D]
        t_embeddings = self.time_embeddings(self.t_schedule)

        embedded_sample_trajectory = []
        early_x0_preds = []
        all_predicted_residual = []

        for t_idx in range(1, self.T + 1):

            t_embedding = t_embeddings[-t_idx][None]

            predicted_residual = self.residual_model(base_samples, t_embedding, input_vector)

            #### XXX CHECK IF DYNAMICS ACTUALLY WORK
            # true_x0 = input_vector[...,:2].float()
            # actual_residual = ((base_samples @ self.auxiliary_embedding_matrix.T) - (self.a_t_schedule[-t_idx] * true_x0)) / self.root_b_t_schedule[-t_idx]
            # predicted_residual = predicted_residual - (predicted_residual @ self.sample_subspace_accessor) + (actual_residual @ self.auxiliary_embedding_matrix)
            ####

            base_samples, early_embedded_x0_pred = self.denoise_one_step(t_idx, base_samples, predicted_residual, noise_scaler)
            early_x0_pred = early_embedded_x0_pred @ self.auxiliary_embedding_matrix.T      # [..., 1, sample dim]

            embedded_sample_trajectory.append(base_samples.detach())
            early_x0_preds.append(early_x0_pred.detach())
            all_predicted_residual.append(predicted_residual.detach())
        
        embedded_sample_trajectory = torch.concat(embedded_sample_trajectory, -2) # [..., T, sample_ambient_dim]
        sample_trajectory = embedded_sample_trajectory @ self.auxiliary_embedding_matrix.T  # [..., T, dim x]
        early_x0_preds = torch.concat(early_x0_preds, -2) # [..., T, dim x]
        all_predicted_residual = torch.concat(all_predicted_residual, -2) @ self.auxiliary_embedding_matrix.T
        new_samples = (base_samples.squeeze(-2).detach() @ self.auxiliary_embedding_matrix.T)

        return {
            'sample_trajectory': sample_trajectory.cpu(),
            'samples': new_samples.detach().cpu(),
            'early_x0_preds': early_x0_preds.cpu(),
            'epsilon_hat': all_predicted_residual.detach().cpu(),
        }



class RNNBaselineDDPMReverseProcess(LinearSubspaceTeacherForcedDDPMReverseProcess):
    """
    Just a baseline, noiseless baseline to see if the dynamics of the network are good
    enough to do the most basic task(s) - e.g. a unimodal distribution

    Generation is the same as LinearSubspaceTeacherForcedDDPMReverseProcess, since no teacher-forcing is done there,
        but residual estimation is a bit of hack - it's not actually residual estimation - see below!

    To go even even closer to the baseline, use_leaky will abandon the standard scaling of noise, old state, and non-linear
        output in favour of the standard leaky RNN dynamics, eith euler alpha = 0.1
    """

    def __init__(self, seperate_output_neurons: bool, use_leaky: bool, stabilise_nullspace: bool, sample_ambient_dim: int, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, device='cuda') -> None:
        super().__init__(seperate_output_neurons, stabilise_nullspace, sample_ambient_dim, sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, device)

        self.do_teacher_forcing = False

        assert not (stabilise_nullspace and use_leaky)

        if use_leaky:
            self.noise_scaler_schedule[1:] = torch.ones_like(self.noise_scaler_schedule[1:]) * self.noise_scaler_schedule[1]
            self.base_samples_scaler_schedule = torch.ones_like(self.base_samples_scaler_schedule) * 0.9        # like 1 - euler_alpha
            self.residual_scaler_schedule = 1.0 - self.base_samples_scaler_schedule

    def noise(self, x_0: _T) -> Dict[str, _T]:
        """
        x_0 of shape [..., <shape x>]

        **Slightly hacking the system here**
            
            x_t = nothing - no changes made to x_0 except to repeat on time dimension --> [..., T, <shape x>]
            epsilon = same as x_t!!! This will be fed to ExampleSampleGenerator.mse alongside the outputs of
                the RNNBaselineDDPMReverseProcess.residual, and in this case we just do vanilla mse
        """
        assert list(x_0.shape[-len(self.sample_shape):]) == self.sample_shape, f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        num_extra_dim = len(x_0.shape) - len(self.sample_shape)
        x_t = x_0.unsqueeze(-len(self.sample_shape)-1)
        x_t = x_t.expand(*x_0.shape[:num_extra_dim], self.T, *self.sample_shape)

        return {
            'x_t': x_t,
            'epsilon': x_t,
        }

    def residual(self, x_samples: _T, network_input: _T, initial_state: Optional[_T] = None, kwargs_for_residual_model = {}) -> Dict[str, _T]:
        """
        **Again, hacking the system here**

        x_samples of shape [..., T, <shape x>] but not used at all here!
        network_input of shape [..., <shape Z>] - as in LinearSubspaceTeacherForcedDDPMReverseProcess
        initial_state of shape [..., ambient space dim] - as in LinearSubspaceTeacherForcedDDPMReverseProcess

        This is the RNN baseline class:
            - No teacher forcing
            - output no longer predicts residuals (epsilon_hat above) but gives the network trajectory in the linear subspace of the sames
                NB: only intended to be used for the simplest tasks!
        """
        return super(RNNBaselineDDPMReverseProcess, self).residual(
            x_samples=x_samples, network_input=network_input, initial_state=initial_state, **kwargs_for_residual_model
        )



class PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(LinearSubspaceTeacherForcedDDPMReverseProcess):

    def __init__(self, num_prep_steps: int, network_input_during_diffusion: bool, seperate_output_neurons: bool, stabilise_nullspace: bool, sample_ambient_dim: int, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, device='cuda') -> None:

        super().__init__(seperate_output_neurons, stabilise_nullspace, sample_ambient_dim, sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, device)

        self.register_parameter('prep_time_embedding', nn.Parameter(torch.randn(1, time_embedding_size), requires_grad = True))
        self.network_input_during_diffusion = network_input_during_diffusion
        self.num_prep_steps = num_prep_steps


    def prepare(self, network_input: _T, batch_shape: List[int], kwargs_for_residual_model = {}) -> Dict[str, _T]:
        """
        Generate initial states to feed into residual or generate_samples

        As with those methods, input_vector of shape [..., <shape Z>], where [...] given by batch_shape
            Won't check this here as they will be checked downstream
        """
        
        initial_state = torch.randn(*batch_shape, 1, self.sample_ambient_dim, device = self.sigma2xt_schedule.device) * self.base_std # [..., 1, D]
        input_vector = self.input_model(network_input).unsqueeze(-2)         # [..., 1, Di]

        recent_state = initial_state
        preparatory_trajectory = []

        for t_idx in range(1, self.num_prep_steps):
            
            # NB: this is not actually a residual!
            embedded_predicted_residual = self.residual_model(recent_state, self.prep_time_embedding, input_vector)
            recent_state, _ = self.denoise_one_step(t_idx, recent_state, embedded_predicted_residual, noise_scaler=1.0)
            preparatory_trajectory.append(recent_state)


        preparatory_trajectory = torch.concat(preparatory_trajectory[::-1], -2)        # Switched to forward time time!

        return {
            'preparatory_trajectory': preparatory_trajectory,
            'postprep_state': preparatory_trajectory[...,0,:],    # Again, forward time, so first state here will be first state for the denoising
            'postprep_base_samples': preparatory_trajectory[...,0,:] @ self.auxiliary_embedding_matrix.T,
        }

    def residual(self, x_samples: _T, network_input: _T, kwargs_for_residual_model={}) -> Dict[str, _T]:
        prep_dict = self.prepare(network_input, x_samples.shape[:-2])
        network_input_mult = 1.0 if self.network_input_during_diffusion else 0.0
        residual_dict = super().residual(x_samples, network_input * network_input_mult, prep_dict['postprep_state'], kwargs_for_residual_model)
        return dict(**prep_dict, **residual_dict)

    def generate_samples(self, network_input: _T, samples_shape: List[int], noise_scaler: float = 1.0, kwargs_for_residual_model={}) -> Dict[str, _T]:
        prep_dict = self.prepare(network_input, samples_shape)
        network_input_mult = 1.0 if self.network_input_during_diffusion else 0.0
        samples_dict = super().generate_samples(network_input * network_input_mult, None, prep_dict['postprep_state'], noise_scaler, kwargs_for_residual_model)
        return dict(**prep_dict, **samples_dict)


