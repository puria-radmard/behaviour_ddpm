from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor as _T

from ddpm.model.residual import VectoralResidualModel
from ddpm.model.input import InputModelBlock

from sampling_ddpm.ddpm.model.main.base import PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess


class MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
    PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
):
    """
    This is for tasks which have multiple (potenitally variable number of) epochs, the last of which is diffusion
    If we want to mask input during the diffusion stage, we have to specify that
    The model class stores nothing about number of epochs or their durations - this are all passed to self.residual and self.generate_samples
    """

    def __init__(
        self,
        seperate_output_neurons: bool,
        stabilise_nullspace: bool,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device="cuda",
    ) -> None:
        super().__init__(
            None,
            None,
            seperate_output_neurons,
            stabilise_nullspace,
            sample_ambient_dim,
            sample_shape,
            sigma2xt_schedule,
            residual_model,
            input_model,
            time_embedding_size,
            device,
        )

        # Just to be sure
        del self.num_prep_steps
        del self.network_input_during_diffusion

    def residual(
        self,
        x_samples: _T,
        prep_network_inputs: List[_T],
        diffusion_network_inputs: List[_T],
        prep_epoch_durations: List[int],
        diffusion_epoch_durations: List[Optional[int]],
        kwargs_for_residual_model={},
        override_initial_state: Optional[_T] = None
    ) -> Tuple[List[Dict[str, _T]], Dict[str, _T]]:
        assert len(prep_network_inputs) == len(prep_epoch_durations)
        all_prep_dicts = [
            self.prepare(
                prep_network_inputs[0], x_samples.shape[:-2], prep_epoch_durations[0], override_initial_state=override_initial_state
            )
        ]
        for pni, ped in zip(prep_network_inputs[1:], prep_epoch_durations[1:]):
            all_prep_dicts.append(
                self.prepare(
                    pni,
                    x_samples.shape[:-2],
                    ped,
                    override_initial_state=all_prep_dicts[-1]["postprep_state"],
                )
            )
        if len(diffusion_epoch_durations) == 1:
            assert diffusion_epoch_durations[0] == None
            residual_dict = super(
                PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess, self
            ).residual(
                x_samples,
                diffusion_network_inputs[0],
                all_prep_dicts[-1]["postprep_state"],
                kwargs_for_residual_model,
            )
            return all_prep_dicts, residual_dict
        else:
            assert sum(diffusion_epoch_durations) == self.T, \
                "Tasks with multiple diffusion epochs need to have sum(diffusion_epoch_durations) == self.T"
            raise NotImplementedError('Multiple diffusion epochs generation not sorted out yet!')

    def generate_samples(
        self,
        *_,
        prep_network_inputs: List[_T],
        diffusion_network_inputs: List[_T],
        prep_epoch_durations: List[int],
        diffusion_epoch_durations: List[Optional[int]],
        samples_shape: List[int],
        noise_scaler: float = 1.0,
        kwargs_for_residual_model={},
        end_t_idx=None,
        override_initial_state: Optional[_T] = None
    ) -> Tuple[List[Dict[str, _T]], Dict[str, _T]]:
        assert len(prep_network_inputs) == len(prep_epoch_durations)
        all_prep_dicts = [
            self.prepare(prep_network_inputs[0], samples_shape, prep_epoch_durations[0], override_initial_state = override_initial_state)
        ]
        for pni, ped in zip(prep_network_inputs[1:], prep_epoch_durations[1:]):
            all_prep_dicts.append(
                self.prepare(
                    pni,
                    samples_shape,
                    ped,
                    override_initial_state=all_prep_dicts[-1]["postprep_state"],
                )
            )
        if len(diffusion_epoch_durations) == 1:
            assert diffusion_epoch_durations[0] == None
            diffusion_epoch_durations = [end_t_idx]
            samples_dict = super(
                PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess, self
            ).generate_samples(
                network_input=diffusion_network_inputs[0],
                samples_shape=None,
                base_samples=all_prep_dicts[-1]["postprep_state"],
                noise_scaler=noise_scaler,
                kwargs_for_residual_model=kwargs_for_residual_model,
                start_t_idx=1,
                end_t_idx=end_t_idx,
            )
            return all_prep_dicts, samples_dict
        else:
            assert (end_t_idx == None) and sum(diffusion_epoch_durations) == self.T, \
                "Tasks with multiple diffusion epochs need to have sum(diffusion_epoch_durations) == self.T. You also cannot override end_t_idx = None during generation"
            raise NotImplementedError('Multiple diffusion epochs generation not sorted out yet!')


class InitialisedSampleSpacePreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
):
    """
    Same as before except you have the option to also override the sample space activity
    before the first preparatory epoch

    Namely, tasks with InitialisedSampleSpaceTaskVariableGenerator as the task variable generator

    Desired pattern is given in MultiepochTrialInformation.pre_prep_sample_information
    """

    def __init__(self, seperate_output_neurons: bool, stabilise_nullspace: bool, sample_ambient_dim: int, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, device="cuda") -> None:
        super().__init__(seperate_output_neurons, stabilise_nullspace, sample_ambient_dim, sample_shape, sigma2xt_schedule, residual_model, input_model, time_embedding_size, device)

    def random_state_from_samples(self, pre_prep_samples: _T) -> _T:
        import pdb; pdb.set_trace()

        num_extra_dim = (
            len(pre_prep_samples.shape) - len(self.sample_shape) - 1
        )  # including time now
        batch_shape = pre_prep_samples.shape[:num_extra_dim]

        initial_state = (
            torch.randn(
                *batch_shape,
                1,
                self.sample_ambient_dim,
                device=self.sigma2xt_schedule.device,
            )
            * self.base_std
        )  # [..., 1, D]

        embedded_samples = (
            pre_prep_samples @ self.auxiliary_embedding_matrix
        )

        sample_removed_initial_state = initial_state - (
            initial_state @ self.sample_subspace_accessor
        )

        sample_injected_initial_state = sample_removed_initial_state + embedded_samples

        return sample_injected_initial_state

    def residual(self, x_samples: _T, prep_network_inputs: List[_T], diffusion_network_inputs: List[_T], prep_epoch_durations: List[int], diffusion_epoch_durations: List[Optional[int]], pre_prep_samples: _T, kwargs_for_residual_model={}) -> Tuple[List[Dict[str, _T]], Dict[str, _T]]:
        override_initial_state = self.random_state_from_samples(pre_prep_samples)
        return super().residual(x_samples, prep_network_inputs, diffusion_network_inputs, prep_epoch_durations, diffusion_epoch_durations, kwargs_for_residual_model, override_initial_state)

    def generate_samples(self, *_, prep_network_inputs: List[_T], diffusion_network_inputs: List[_T], prep_epoch_durations: List[int], diffusion_epoch_durations: List[Optional[int]], samples_shape: List[int], pre_prep_samples: _T, noise_scaler: float = 1, kwargs_for_residual_model={}, end_t_idx=None) -> Tuple[List[Dict[str, _T]], Dict[str, _T]]:
        override_initial_state = self.random_state_from_samples(pre_prep_samples)
        return super().generate_samples(*_, prep_network_inputs=prep_network_inputs, diffusion_network_inputs=diffusion_network_inputs, prep_epoch_durations=prep_epoch_durations, diffusion_epoch_durations=diffusion_epoch_durations, samples_shape=samples_shape, noise_scaler=noise_scaler, kwargs_for_residual_model=kwargs_for_residual_model, end_t_idx=end_t_idx, override_initial_state=override_initial_state)
