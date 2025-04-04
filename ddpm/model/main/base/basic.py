from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Dict, Optional, List, Tuple

from abc import ABC, abstractmethod

from ddpm.model.embedding_reprs import TimeEmbeddingBlock
from ddpm.model.transition import VectoralResidualModel, UNetResidualModel
from ddpm.model.input import InputModelBlock

from ddpm.tasks.distribution import DistributionInformation

from typing import List, Mapping, Any


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
        device="cuda",
    ) -> None:
        super(DDPMReverseProcessBase, self).__init__()

        self.sample_shape = sample_shape

        self.T = len(sigma2xt_schedule)

        self.residual_model = residual_model
        self.time_embeddings = TimeEmbeddingBlock(
            self.T, time_embedding_size, device=device
        )
        self.input_model = input_model
        assert (
            input_model.network_input_size == residual_model.input_size
        ), f"Got an input model (sensory -> residual network) with output size {input_model.network_input_size}, but a residual model that accepts inputs of size {residual_model.input_size}"

        # Incremental noising
        self.sigma2xt_schedule = sigma2xt_schedule  # t = 1, ..., T
        assert len(sigma2xt_schedule.shape) == 1
        self.t_schedule = torch.arange(self.T)  # Indexes self.time_embeddings.time_embs
        self.incremental_modulation_schedule = torch.sqrt(1.0 - self.sigma2xt_schedule)
        self.std_schedule = torch.sqrt(self.sigma2xt_schedule)

        # One-shot noising
        self.a_t_schedule = torch.cumprod(
            self.incremental_modulation_schedule, 0
        )  # t = 1, ..., T
        self.root_b_t_schedule = torch.sqrt(
            1.0 - torch.square(self.a_t_schedule)
        )  # t = 1, ..., T

        # Generation
        self.base_std = self.root_b_t_schedule[-1]  # scalar, final variance = b_T
        noise_scaler_schedule = (
            self.root_b_t_schedule[:-1]
            / self.root_b_t_schedule[1:]
            * self.std_schedule[1:]
        )  # t = 2, ..., T
        self.noise_scaler_schedule = torch.concat(
            [
                torch.zeros(1, device=noise_scaler_schedule.device),
                noise_scaler_schedule,
            ],
            0,
        )  # Still t = 2, ..., T but with a zero appended to the front for ease!
        self.base_samples_scaler_schedule = (
            1.0 / self.incremental_modulation_schedule
        )  # t = 1, ..., T
        self.residual_scaler_schedule = (
            self.sigma2xt_schedule
            / self.incremental_modulation_schedule
            / self.root_b_t_schedule
        )

        ## Training
        # mse_scaler_schedule_num = (1 - self.a_t_schedule[1:] ** 2.0) * self.sigma2xt_schedule[1:]    # t = 2, ..., T
        # mse_scaler_schedule_denom = 2 * (1 - self.a_t_schedule[:-1] ** 2.0) * (1.0 - self.sigma2xt_schedule[1:])    # t = 2, ..., T
        # self.mse_scaler_schedule = mse_scaler_schedule_num / mse_scaler_schedule_denom

        ## Convinience
        self.reshaped_a_t_schedule = self.a_t_schedule[
            :, *[None for _ in self.sample_shape]
        ]  # [T, <1 for each of shape x>]
        self.reshaped_root_b_t_schedule = self.root_b_t_schedule[
            :, *[None for _ in self.sample_shape]
        ]  # [T, <1 for each of shape x>]

        self.reshaped_base_samples_scaler_schedule = self.base_samples_scaler_schedule[
            :, *[None for _ in self.sample_shape]
        ]
        self.reshaped_residual_scaler_schedule = self.residual_scaler_schedule[
            :, *[None for _ in self.sample_shape]
        ]

        self.device = device
        self.to(device)

    def to(self, *args, **kwargs):
        self.sigma2xt_schedule = self.sigma2xt_schedule.to(*args, **kwargs)
        self.t_schedule = self.t_schedule.to(*args, **kwargs).int()
        self.incremental_modulation_schedule = self.incremental_modulation_schedule.to(
            *args, **kwargs
        )
        self.a_t_schedule = self.a_t_schedule.to(*args, **kwargs)
        self.reshaped_a_t_schedule = self.reshaped_a_t_schedule.to(*args, **kwargs)
        self.root_b_t_schedule = self.root_b_t_schedule.to(*args, **kwargs)
        self.reshaped_root_b_t_schedule = self.reshaped_root_b_t_schedule.to(
            *args, **kwargs
        )
        self.reshaped_base_samples_scaler_schedule = self.reshaped_base_samples_scaler_schedule.to(
            *args, **kwargs
        )
        self.reshaped_residual_scaler_schedule = self.reshaped_residual_scaler_schedule.to(
            *args, **kwargs
        )
        # self.mse_scaler_schedule = self.mse_scaler_schedule.to(*args, **kwargs)
        self.noise_scaler_schedule = self.noise_scaler_schedule.to(*args, **kwargs)
        return super(DDPMReverseProcessBase, self).to(*args, **kwargs)

    @abstractmethod
    def noise(self, x_0: _T) -> Dict[str, _T | int]:
        """
        x_0 of shape [..., <shape x>]

        Requires output contains at least x_t and epsilon, with standard definitions given in OneShotDDPMReverseProcess.noise
        """
        raise NotImplementedError

    @abstractmethod
    def residual(
        self, x_samples: _T, network_input: _T, **kwargs_for_residual_model
    ) -> Dict[str, _T]:  # epsilon: Optional[_T] = None,
        """
        Given a full trajectory of x_t, predict the *one-shot* epsilons that produced them
        In the teacher-forcing case, we are actually predicting the *effective* epsilons that would have cause such a one-shot noising pattern
        """
        raise NotImplementedError

    def denoise_one_step(
        self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float
    ):
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
        scaled_residual = (
            self.residual_scaler_schedule[-t_idx] * predicted_residual
        )  # [..., 1, dim x]

        one_step_denoise = scaled_base_samples - scaled_residual + scaled_noise
        early_x0_pred = (
            x_t_plus_1 - self.root_b_t_schedule[-t_idx] * predicted_residual
        ) / self.a_t_schedule[-t_idx]

        return one_step_denoise, early_x0_pred

    @abstractmethod
    def generate_samples(
        self,
        *_,
        network_input: _T,
        samples_shape: Optional[List[int]] = None,
        base_samples: Optional[_T] = None,
        noise_scaler: float = 1.0,
    ) -> Dict[str, _T]:
        """
        input_vector of shape [..., <shape Z>]
        If provided, base_samples of shape [..., <shape x>]
        Otherwise, give ... = samples_shape

        sample_trajectory of shape [..., T, <shape x>]
        samples of shape [..., <shape x>]
        early_x0_preds of shape [..., T, <shape x>]
        """
        raise NotImplementedError

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        *_,
        kept_input_dims: Optional[List[int]] = None,
    ):
        try:
            ret = super().load_state_dict(state_dict, strict, assign)
            if kept_input_dims is not None:
                print(
                    "warning: used kept_input_dims in DDPMReverseProcessBase.load_state_dict even though state dict fit fine"
                )
            return ret

        except RuntimeError as e:
            assert not assign
            assert strict
            assert kept_input_dims is not None
            print(e)
            print("Trying again without input weights!")
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name == "residual_model.layers.0.weight":
                    num_rec_and_time_inds = (
                        self.residual_model.state_space_size
                        + self.time_embeddings.time_embedding_dim
                    )
                    inputs_dim = [ki + num_rec_and_time_inds for ki in kept_input_dims]
                    required_dims = list(range(num_rec_and_time_inds)) + inputs_dim
                    copyable_param = param.data[:, required_dims]
                else:
                    copyable_param = param.data
                try:
                    own_state[name].copy_(copyable_param)
                except RuntimeError as e2:
                    print(e2)
                    print("Copying weight in with random input weights!")
                    own_state[name][:, : len(required_dims)] = copyable_param



class OneShotDDPMReverseProcess(DDPMReverseProcessBase):
    """
    Standard DDPM noising - whole space noised together, space noised one-shot

    At training time, residuals are predicted for each timestep independently
    """

    @torch.no_grad()
    def noise(self, x_0: _T) -> Dict[str, _T | int]:
        """
        x_0 of shape [..., <shape x>]

        Both outputs of shape [..., T, dim x]
            x_t = one-shot noising
            epsilon = actual epsilon used for one-shot noising
        """
        assert (
            list(x_0.shape[-len(self.sample_shape) :]) == self.sample_shape
        ), f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        num_extra_dim = len(x_0.shape) - len(self.sample_shape)
        x_0 = x_0.unsqueeze(num_extra_dim).expand(
            *x_0.shape[:num_extra_dim],
            self.T,
            *self.sample_shape,
            # *[1] * num_extra_dim, self.T, *[1] * len(self.sample_shape)
        )
        epsilon = torch.randn_like(x_0)

        # [..., T, dim x]
        x_t_means = self.reshaped_a_t_schedule[*[None] * num_extra_dim] * x_0
        x_t = (
            x_t_means
            + self.reshaped_root_b_t_schedule[*[None] * num_extra_dim] * epsilon
        )

        return {"x_t": x_t, "epsilon": epsilon}

    def residual(
        self, x_samples: _T, network_input: _T, kwargs_for_residual_model={}
    ) -> Dict[str, _T]:
        """
        x_samples of shape [..., T, <shape x>]
            Forward time (1 -> T) so increasingly noised
        network_input of shape [..., <shape Z>]       --> same for all timesteps in this class!
        """
        num_extra_dim = (
            len(x_samples.shape) - len(self.sample_shape) - 1
        )  # including time now
        batch_shape = x_samples.shape[:num_extra_dim]

        # input_vectors generated in forward time (1 --> T), so can just be plugged in
        input_vectors: _T = self.input_model(network_input, self.T)
        assert tuple(input_vectors.shape) == (
            *batch_shape,
            self.T,
            self.residual_model.input_size,
        ), f"Expected input_vector shape to be {(*batch_shape, self.T, self.residual_model.input_size)} but got {tuple(input_vectors.shape)}"

        assert tuple(x_samples.shape) == (
            *batch_shape,
            self.T,
            *self.sample_shape,
        ), f"Expected x_samples shape to end in {self.sample_shape} but got {x_samples.shape}"

        t_embeddings = self.time_embeddings(self.t_schedule)
        epsilon_hat = self.residual_model(
            x=x_samples,
            t_embeddings_schedule=t_embeddings,
            input_vector=input_vectors,
            **kwargs_for_residual_model,
        )  # [..., T, <sample shape>]
        assert x_samples.shape == epsilon_hat.shape

        return {"epsilon_hat": epsilon_hat}

    def generate_samples(
        self,
        *_,
        network_input: _T,
        samples_shape: Optional[List[int]] = None,
        base_samples: Optional[_T] = None,
        noise_scaler: float = 1.0,
    ) -> Dict[str, _T]:
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
            base_samples = (
                torch.randn(
                    *samples_shape,
                    *self.sample_shape,
                    device=self.sigma2xt_schedule.device,
                )
                * self.base_std
            )
        else:
            samples_shape = base_samples.shape[: -len(self.sample_shape)]
            assert tuple(base_samples.shape) == (*samples_shape, *self.sample_shape)

        input_vectors = self.input_model(
            network_input, self.T
        )  # forward time so need to be indexed backwards

        assert tuple(input_vectors.shape) == (
            *samples_shape,
            self.T,
            self.residual_model.input_size,
        ), f"Expected input_vector shape to be {(*samples_shape, self.T, self.residual_model.input_size)} but got {tuple(input_vectors.shape)}"

        base_samples = base_samples.unsqueeze(len(samples_shape))  # [..., 1, D]
        t_embeddings = self.time_embeddings(self.t_schedule)

        sample_trajectory = []
        early_x0_preds = []

        for t_idx in range(1, self.T + 1):

            t_embedding = t_embeddings[-t_idx][None]  # [1, Dt]

            predicted_residual = self.residual_model(
                base_samples,
                t_embedding,
                input_vectors[..., [-t_idx], :],  # TODO: fix this with a method!
            )

            base_samples, early_x0_pred = self.denoise_one_step(
                t_idx, base_samples, predicted_residual, noise_scaler
            )

            sample_trajectory.append(base_samples.detach().cpu())
            early_x0_preds.append(early_x0_pred.detach().cpu())

        sample_trajectory = torch.concat(
            sample_trajectory, len(samples_shape)
        )  # [..., T, <shape x>]
        early_x0_preds = torch.concat(
            early_x0_preds, len(samples_shape)
        )  # [..., T, <shape x>]

        return {
            "sample_trajectory": sample_trajectory,
            "samples": base_samples.squeeze(-2),
            "early_x0_preds": early_x0_preds,
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

    def __init__(
        self,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device="cuda",
    ) -> None:
        super().__init__(
            sample_shape,
            sigma2xt_schedule,
            residual_model,
            input_model,
            time_embedding_size,
            device,
        )

    @torch.no_grad()
    def noise(self, x_0: _T) -> Dict[str, _T | int]:
        """
        x_0 of shape [..., <shape x>]

        All outputs of shape [..., T, dim x]
            x_t = noising trajectory (actually a trajectory here)
            epsilon_actual = actual incrementental epsilons added to samples (before scaling and modulation)
            epsilon = effective epsilon, which would have taken us from x_0 to x_t one shot (after scaling and modulation)
        """
        assert (
            list(x_0.shape[-len(self.sample_shape) :]) == self.sample_shape
        ), f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        epsilon_actual = torch.randn(self.T, *x_0.shape).to(x_0.device)
        num_extra_dim = len(x_0.shape) - len(self.sample_shape)

        # Construct trajectory
        trajectory = [x_0]
        for t, eps in enumerate(epsilon_actual):  # Forward time
            trajectory.append(
                (trajectory[-1] * self.incremental_modulation_schedule[t])
                + (self.std_schedule[t] * eps)
            )
        x_t = torch.stack(trajectory[1:], num_extra_dim)

        # Effective epsilons, which should be predicted
        epsilon_effective = (
            x_t - (self.reshaped_a_t_schedule * x_0.unsqueeze(num_extra_dim))
        ) / self.reshaped_root_b_t_schedule

        return {
            "x_t": x_t,
            "epsilon_actual": epsilon_actual,
            "epsilon": epsilon_effective,
            "num_extra_dim": num_extra_dim
        }

