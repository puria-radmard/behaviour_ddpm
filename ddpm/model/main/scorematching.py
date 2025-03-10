import torch
from torch import nn
from torch import Tensor as _T

from typing import Dict, List, Optional, Tuple

from ddpm.model.main.base import DDPMReverseProcessBase, LinearSubspaceTeacherForcedDDPMReverseProcess, PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
from ddpm.model.main.multiepoch import MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess

from ddpm.tasks.distribution import DistributionInformation

from abc import ABC, abstractmethod


class ScoreMatchingHelper(DDPMReverseProcessBase, ABC):
    """
    DDPMReverseProcessBase is based on the DDPM design which works with sample trajectories

    This is intended for closed form distributions over data q(x_0), 
        such that we can extract the real marginal q(x_t) = <q(x_t | x_0)>_{q(x_0)}

    The main (pretty much only) reason to keep inheritance from DDPMReverseProcessBase is to maintain
        the relevant DDPMReverseProcessBase.generate_samples functions

        In the multiepoch case, we also want to inherit preparatory activity, which is effectively
        the same as sample generation

        (and to not have to redefine noising schedules etc.)
    
    Training is not longer done with DDPMReverseProcessBase.residual - this will raise an error
        Similarily, DDPMReverseProcessBase.noise has been overridden with an error

    Training is now done with ScoreMatchingHelper.est_score
        This will start with some base samples, and run reverse-time dynamics as usual
        
        Rather than x_samples to do teacher forcing, it will accept a DistributionInformation
            This is because the whole point of switching to closed form scores was to not have to limit learning
            to some small set of samples, which may not be rich enough for a large number of modes
        
            DistributionInformation instead gives us the true score (for the sample space), which
            is used to do (XXX: weighted) teacher forcing
        
        For prep/multiepoch models, there will be the option(XXX) to reinitialise sample-space activity
            at the base distribution

    XXX: TODO: write instruction on __mro__ here!
    """

    def denoise_one_step(
        self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float
    ):
        """
        t_idx indexes time backwards, so ranges from 1 to T

        x_t_plus_1 is samples denoised so far - of shape [..., 1, <shape x>]
        predicted_residual is now the estimated score at the point x_t_plus_1, of the same size

        We are using this method to override the actual dynamics of the denoising process, which
            is slightly different when the output of the residual model is actually the estimated score at the point x_t_plus_1

        early_x0_pred also no longer has a meaning, so we return it as NaNs
        """

        assert t_idx > 0 and t_idx <= self.T

        # assert list(x_t_plus_1.shape) == [num_samples, 1, *self.sample_shape] == list(predicted_residual.shape)
        noise = noise_scaler * torch.randn_like(x_t_plus_1)
        scaled_noise = noise * self.std_schedule[-t_idx]

        scaled_base_samples = x_t_plus_1 / self.incremental_modulation_schedule[-t_idx]
        scaled_score = (
            self.sigma2xt_schedule[-t_idx] * predicted_residual / self.incremental_modulation_schedule[-t_idx]
        )  # [..., 1, dim x]

        one_step_denoise = scaled_score + scaled_base_samples + scaled_noise
        
        early_x0_pred = torch.ones_like(predicted_residual) * torch.nan

        return one_step_denoise, early_x0_pred

    def noise(self, x_0: _T):
        raise TypeError('ScoreMatchingHelper subclass cannot use DDPMReverseProcessBase.noise')

    def residual(self, x_samples: _T, network_input: _T, **kwargs_for_residual_model):
        raise TypeError('ScoreMatchingHelper subclass cannot use DDPMReverseProcessBase.residual')
    
    @abstractmethod
    def est_score(self, target_distribution: DistributionInformation, network_input: _T, **kwargs_for_residual_model) -> Dict[str, _T]:
        raise NotImplementedError
    




class ScoreMatchingLinearSubspaceTeacherForcedDDPMReverseProcess(ScoreMatchingHelper, LinearSubspaceTeacherForcedDDPMReverseProcess):
    
    def est_score(self, target_distribution: DistributionInformation, network_input: _T, initial_state: Optional[_T] = None, kwargs_for_residual_model={}) -> Dict[str, _T]:
        """
        network_input of shape [..., <shape Z>]
            still the same for all timesteps, if the network is to be used for computation beforehand, it can should do so
            before calling this method, i.e. encoded in initial_state - see below
        
        target_distribution.batch_shape should broadcast predictably with [...] and match sample_shape with self

        initial_state of shape [..., ambient space dim]
            starting combined state of sample variable and auxiliary computation variables, all embedded into the larger ambient space
        """
        
        #for mroi in self.__class__.__mro__:
        #    print(mroi)

        input_vectors: _T = self.input_model(network_input, self.T)
        num_extra_dim = (
            len(input_vectors.shape) - len(self.sample_shape) - 1
        )  # including time, and basing it on network_input now, instead of x_samples, which are not given here
        batch_shape = input_vectors.shape[:num_extra_dim]

        assert tuple(input_vectors.shape) == (
            *batch_shape,
            self.T,
            self.residual_model.input_size,
        ), f"Expected input_vector shape to be {(*batch_shape, self.T, self.residual_model.input_size)} but got {tuple(input_vectors.shape)}"

        assert (num_extra_dim == target_distribution.num_extra_dims) == len(torch.broadcast_shapes(target_distribution.num_extra_dims, num_extra_dim)), \
            f"Provided network_input of batch shape {batch_shape} but target_distribution expects batch shape {target_distribution.num_extra_dims} - these do not broadcast"
        
        assert target_distribution.sample_shape == self.sample_shape
        
        t_embeddings = self.time_embeddings(self.t_schedule)

        if initial_state is None:
            initial_state = (
                torch.randn(
                    *batch_shape,
                    self.sample_ambient_dim,
                    device=self.sigma2xt_schedule.device,
                )
                * self.base_std
            )
        else:
            assert tuple(initial_state.shape) == (
                *batch_shape,
                self.sample_ambient_dim,
            ), f"Expected initial_state shape to end with {self.sample_ambient_dim} but got {tuple(initial_state.shape)}"

        one_step_denoising = initial_state.unsqueeze(-2)  # [..., 1, ambient space dim]
        all_predicted_scores = []
        all_subspace_trajectories = []

        if self.do_teacher_forcing:

            subspace_activity = torch.randn(*batch_shape, 1, *self.sample_shape, dtype=one_step_denoising.dtype, device=one_step_denoising.device)
            
            # Start off in high pdf location
            one_step_denoising = one_step_denoising - (
                one_step_denoising @ self.sample_subspace_accessor
            ) + (
                subspace_activity @ self.auxiliary_embedding_matrix
            )


        for t_idx in range(1, self.T + 1):

            # Denoise in the full ambient space for one step: one_step_denoising, early_embedded_x0_pred both of shape [..., 1, ambient space dim]
            # Unlike before (in teacher forced DDPM case), we hold off committing to changes for a moment...
            t_embedding = t_embeddings[-t_idx][None]
            embedded_score_estimation = self.residual_model(
                one_step_denoising, t_embedding, input_vectors[..., [-t_idx], :]
            )

            subspace_activity = one_step_denoising @ self.auxiliary_embedding_matrix.T
            all_subspace_trajectories.append(subspace_activity)

            # If required, correct the state
            # Unlike before (in teacher forced DDPM case), we actually have to calculate the correct rollout first...
            if self.do_teacher_forcing:

                true_score = target_distribution.calculate_score(
                    subspace_activity, self.a_t_schedule[[-t_idx]], self.root_b_t_schedule[[-t_idx]].square()
                ).detach() # [..., 1, sample_size]

                true_subspace_next_step, _ = self.denoise_one_step(
                    t_idx, subspace_activity, true_score, noise_scaler=1.0
                ) # [..., 1, sample_size]

            # XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ###
            # print('replacing with true score')
            # embedded_score_estimation = embedded_score_estimation - (
            #     embedded_score_estimation @ self.sample_subspace_accessor
            # ) + (
            #     true_score @ self.auxiliary_embedding_matrix
            # )
            # XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ### XXX ###

            one_step_denoising, _ = self.denoise_one_step(
                t_idx, one_step_denoising, embedded_score_estimation, noise_scaler=1.0
            )

            # Only these directions have the actual interpretation of a 'predicted residual'
            predicted_score = (
                embedded_score_estimation @ self.auxiliary_embedding_matrix.T
            )
            all_predicted_scores.append(predicted_score)

            # Overwrite with true rollout
            if self.do_teacher_forcing:

                # Course correct in the linear sample subspace ---> (sample_removed_one_step_denoising @ self.auxiliary_embedding_matrix.T).abs().max() is very small
                one_step_denoising = one_step_denoising - (
                    one_step_denoising @ self.sample_subspace_accessor
                ) + (
                    true_subspace_next_step @ self.auxiliary_embedding_matrix
                )

        score_hat = torch.concat(
            all_predicted_scores[::-1], num_extra_dim
        )  # forward (diffusion) time for downstream MSE loss! i.e. true score will be calculated with the 

        subspace_trajectories = torch.concat(
            all_subspace_trajectories, num_extra_dim
        )  # keep as reverse (denoising) time!

        assert score_hat.shape == subspace_trajectories.shape

        return {
            "score_hat": score_hat,
            "subspace_trajectories": subspace_trajectories,
        }



class ScoreMatchingPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
    ScoreMatchingLinearSubspaceTeacherForcedDDPMReverseProcess,
    PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
):
    
    def est_score(self, target_distribution: DistributionInformation, network_input: _T, **kwargs_for_residual_model) -> Dict[str, _T]:
        batch_shape = network_input.shape[:-len(self.input_model.sensory_shape)]
        prep_dict = self.prepare(
            network_input, batch_shape, self.num_prep_steps
        )
        network_input_mult = 1.0 if self.network_input_during_diffusion else 0.0
        residual_dict = super(ScoreMatchingPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess, self).est_score(
            target_distribution,
            network_input * network_input_mult,
            prep_dict["postprep_state"],
            kwargs_for_residual_model,
        )
        return dict(**prep_dict, **residual_dict)
    

class ScoreMatchingMultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
    ScoreMatchingPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
):

    def est_score(
        self,
        target_distribution: DistributionInformation,
        prep_network_inputs: List[_T],
        diffusion_network_inputs: List[_T],
        prep_epoch_durations: List[int],
        diffusion_epoch_durations: List[Optional[int]],
        kwargs_for_residual_model={},
        override_initial_state: Optional[_T] = None
    ) -> Tuple[List[Dict[str, _T]], Dict[str, _T]]:
        
        # XXX: this should be patched, alongside the assert statement in DiagonalGaussianMixtureDistributionInformation.__init__
        # import pdb; pdb.set_trace(header = 'find a better a way to find out batch_shape here!')
        batch_shape = diffusion_network_inputs[0].shape[:2] # i.e. assume shaoe is [batch, sample, <shape>]

        assert len(prep_network_inputs) == len(prep_epoch_durations)
        all_prep_dicts = [
            self.prepare(
                prep_network_inputs[0], batch_shape, prep_epoch_durations[0], override_initial_state=override_initial_state
            )
        ]
        for pni, ped in zip(prep_network_inputs[1:], prep_epoch_durations[1:]):
            all_prep_dicts.append(
                self.prepare(
                    pni,
                    batch_shape,
                    ped,
                    override_initial_state=all_prep_dicts[-1]["postprep_state"],
                )
            )
        if len(diffusion_epoch_durations) == 1:
            assert diffusion_epoch_durations[0] == None
            residual_dict = super(
                ScoreMatchingPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess, self
            ).est_score(
                target_distribution,
                diffusion_network_inputs[0],
                all_prep_dicts[-1]["postprep_state"],
                kwargs_for_residual_model,
            )
            return all_prep_dicts, residual_dict
        else:
            assert sum(diffusion_epoch_durations) == self.T, \
                "Tasks with multiple diffusion epochs need to have sum(diffusion_epoch_durations) == self.T"
            raise NotImplementedError('Multiple diffusion epochs generation not sorted out yet!')

