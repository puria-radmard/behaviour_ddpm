import math

import torch
from torch import nn
from torch import Tensor as _T

from abc import ABC, abstractmethod

from scipy.stats import norm


class CrossTemporalLinearDecoder(nn.Module):

    def __init__(self, duration, num_neurons, output_size = 2, num_models = 32):
        super().__init__()

        self.duration = duration
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.num_models = num_models

        self.register_parameter('weight', nn.Parameter(torch.randn(num_models, duration, num_neurons, output_size).cuda() / (num_neurons**0.5), requires_grad= True))
        self.register_parameter('bias', nn.Parameter(torch.randn(num_models, duration, output_size).cuda() / output_size, requires_grad= True))

    def decode_sequence(self, sequence: _T, **kwargs) -> _T:
        """
        sequence of shape [batch, duration, neurons]
        output of shape [models, batch, duration, output]
        """
        return torch.einsum('btn,qtnd->qbtd',sequence,self.weight) + self.bias.unsqueeze(1)

    @torch.no_grad()
    def cross_temporal_decoding_inner(self, sequence: _T, **kwargs) -> _T:
        return torch.einsum('btn,qsnd->qbtsd',sequence,self.weight) + self.bias[:,*[None]*2]

    @torch.no_grad()
    def decode_cross_temporally(self, sequence: _T, targets: _T, **kwargs) -> _T:
        """
        sequence of shape [batch, duration, neurons]
        targets of shape [batch, output]

        outputs of shapes:
            decoding    [models, batch, sequence timesteps, decoder timesteps, output]
            losses      [models, batch, sequence timesteps, decoder timesteps]

        Logic recap for bias:
            self.bias [models, duration, output]
            self.bias[:,*[None]*2] [models, 1, 1, duration, output]
            ==> duration axis aligns with decoder timesteps in output
        """
        decoding = self.cross_temporal_decoding_inner(sequence, **kwargs).cpu()

        assert tuple(targets.shape) == (decoding.shape[1], self.output_size), targets.shape
        reshaped_targets = targets.unsqueeze(0).unsqueeze(2).unsqueeze(3).cpu()    # [models, batch, 1, 1, D]
        losses = (reshaped_targets - decoding).square().sum(-1).sqrt()
        
        return decoding, losses



class CrossTemporalNonLinearDecoder(CrossTemporalLinearDecoder):

    def __init__(self, duration, num_neurons, output_size=2, num_models=32):
        super().__init__(duration, num_neurons, output_size, num_models)

        self.register_parameter('weight2', nn.Parameter(torch.randn(num_models, duration, output_size, output_size).cuda() / (num_neurons**0.5), requires_grad= True))
        self.register_parameter('bias2', nn.Parameter(torch.randn(num_models, duration, output_size).cuda() / output_size, requires_grad= True))

    def decode_sequence(self, sequence: _T, **kwargs) -> _T:
        first_layer = torch.nn.functional.softplus(super().decode_sequence(sequence, **kwargs))
        return torch.einsum('qbti,qtij->qbtj',first_layer,self.weight2) + self.bias2.unsqueeze(1)

    @torch.no_grad()
    def cross_temporal_decoding_inner(self, sequence: _T) -> _T:
        first_layer = torch.nn.functional.softplus(super().cross_temporal_decoding_inner(sequence))
        return torch.einsum('qbtsi,qsij->qbtsj',first_layer,self.weight2) + self.bias2[:,*[None]*2]




class CrossTemporalContextGatedDecoder(CrossTemporalLinearDecoder):

    """
    Context gating is across neurons, and we have a different attention mechanism for each timestep.
    """

    def __init__(self, duration, num_neurons, output_size = 2, context_size = 2, num_models = 16) -> None:
        super(CrossTemporalLinearDecoder, self).__init__()

        self.duration = duration
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.context_size = context_size
        self.num_models = num_models

        self.register_parameter('weight_gating_weight', nn.Parameter(torch.randn(num_models, duration, context_size, num_neurons, output_size).cuda() / (num_neurons**0.5), requires_grad=True))
        self.register_parameter('weight_gating_bias', nn.Parameter(torch.randn(num_models, duration, num_neurons, output_size).cuda() / (num_neurons**0.5), requires_grad=True))

        self.register_parameter('bias_gating_weight', nn.Parameter(torch.randn(num_models, duration, context_size, output_size).cuda() / output_size, requires_grad=True))
        self.register_parameter('bias_gating_bias', nn.Parameter(torch.randn(num_models, duration, output_size).cuda() / output_size, requires_grad=True))
    
    def generate_weight(self, context: _T) -> _T:
        """
        context of shape [batch, context size]

        output of shape [models, batch, duration, num neurons, output size]
        """
        contextual_weight = torch.einsum('qtcnd,bc->qbtnd', self.weight_gating_weight, context) + self.weight_gating_bias.unsqueeze(1)
        return contextual_weight

    def generate_bias(self, context: _T) -> _T:
        """
        context of shape [batch, context size]

        output of shape [models, batch, duration, output size]
        """
        contextual_bias = torch.einsum('qtcd,bc->qbtd', self.bias_gating_weight, context) + self.bias_gating_bias.unsqueeze(1)
        return contextual_bias

    def decode_sequence(self, sequence: _T, context: _T) -> _T:
        """
        sequence of shape [batch, duration, neurons]
        context of shape [batch, context size]
        
        output of shape [models, batch, duration, output]
        """
        weight = self.generate_weight(context)
        bias = self.generate_bias(context)

        return torch.einsum('btn,qbtnd->qbtd',sequence,weight) + bias

    @torch.no_grad()
    def cross_temporal_decoding_inner(self, sequence: _T, context: _T) -> _T:
        weight = self.generate_weight(context)
        bias = self.generate_bias(context)

        return torch.einsum('btn,qbsnd->qbtsd',sequence,weight) + bias.unsqueeze(2)




class CrossTemporalNonLinearContextGatedDecoder(CrossTemporalContextGatedDecoder):

    def __init__(self, duration, num_neurons, output_size=2, context_size=2, num_models=16) -> None:
        super().__init__(duration, num_neurons, output_size, context_size, num_models)

        self.register_parameter('weight2', nn.Parameter(torch.randn(num_models, duration, output_size, output_size).cuda() / (num_neurons**0.5), requires_grad= True))
        self.register_parameter('bias2', nn.Parameter(torch.randn(num_models, duration, output_size).cuda() / output_size, requires_grad= True))

    def decode_sequence(self, sequence: _T, context: _T) -> _T:
        first_layer = torch.nn.functional.softplus(super().decode_sequence(sequence, context))
        return torch.einsum('qbti,qtij->qbtj',first_layer,self.weight2) + self.bias2.unsqueeze(1)

    @torch.no_grad()
    def cross_temporal_decoding_inner(self, sequence: _T, context: _T) -> _T:
        first_layer = torch.nn.functional.softplus(super().cross_temporal_decoding_inner(sequence, context))
        return torch.einsum('qbtsi,qsij->qbtsj',first_layer,self.weight2) + self.bias2[:,*[None]*2]





class AllemanStyleRoleReportFeatureProjector(nn.Module, ABC):
    """
    Some form of stimulus_n -> r_n, allowing us to build a mixture model

    In Alleman et al. 2023, this is of the form:
        r(u, l) = Wu f(u) + Wl f(l) + η
    before cuing where:
        u, l are the upper and lower report values
        f is shared spline function (R^2 -> R^K)
        W. are projectors for the upper and lower stimuli,
        and η is noise

    We replace the spline function with a DNN
    
    After cuing, we have:
        r(u,l,c) =  Wlt f(l) + Wud f(u)+η if c=0
                    Wut f(u) + Wld f(l)+η if c=1
    where:
        c = 0 means the lower colour is cued,
        and Wij is the projector for stumulus i when the cue is j (target vs distractor)
    """

    def __init__(self, dim_K: int, dim_R: int, precue_duration: int, postcue_duration: int, shared_p_misbind_over_time: bool, main_layers_sizes = None) -> None:
        super().__init__()

        self.precue_duration = precue_duration
        self.postcue_duration = postcue_duration

        if main_layers_sizes is None:
            main_layers_sizes = [dim_K, dim_K, dim_K]

        self.dim_K = dim_K
        self.dim_R = dim_R

        main_layers = [nn.Linear(2, main_layers_sizes[0]), nn.Sigmoid()]
        for h_in, h_out in zip(main_layers_sizes[:-1], main_layers_sizes[1:]):
            main_layers.extend([nn.Linear(h_in, h_out), nn.Sigmoid()])
        main_layers.extend([nn.Linear(main_layers_sizes[-1], dim_K)])
        self.main_layers = nn.Sequential(*main_layers)

        self.shared_p_misbind_over_time = shared_p_misbind_over_time
        if shared_p_misbind_over_time:
            self.p_misbind_raw = torch.nn.Parameter(0.0 * torch.ones(1).cuda())
            self.p_post_cue_errors_raw = torch.nn.Parameter(0.0 * torch.ones(3).cuda())
        else:
            self.p_misbind_raw = torch.nn.Parameter(0.0 * torch.ones(precue_duration).cuda())
            self.p_post_cue_errors_raw = torch.nn.Parameter(0.0 * torch.ones(precue_duration, 3).cuda())

        self.mode_variance_raw = torch.nn.Parameter(torch.ones(dim_R).cuda())

    @property
    def p_misbind(self) -> _T:
        if self.shared_p_misbind_over_time:
            return self.p_misbind_raw.sigmoid().unsqueeze(0).expand(1, self.precue_duration)
        else:
            return self.p_misbind_raw.sigmoid().unsqueeze(0)

    @property
    def p_post_cue_errors(self) -> _T:
        if self.shared_p_misbind_over_time:
            return self.p_post_cue_errors_raw.softmax(0).unsqueeze(1).unsqueeze(2).expand(3, 1, self.postcue_duration)
        else:
            raise NotImplementedError

    def get_mixture_model_spline_means(self, report_features: _T) -> _T:
        assert len(report_features.shape) == 3, f"Expected report_features of shape [B, N, Dr], got {report_features.shape}"
        return self.main_layers(report_features)

    def get_mixture_model_means_precue(self, report_features: _T, probe_features: _T):
        spline_means = self.get_mixture_model_spline_means(report_features)                 # [B, N, 2] -> [B, N, K]
        gating_matrix = self.generate_gating_matrix(probe_features, 'precued')              # [B, T, N, R, K]
        return torch.einsum('bnk,btnrk->btnr', spline_means, gating_matrix).sum(-2)         # [B, T, R]

    @staticmethod
    def get_gaussian_llh(response: _T, mode_mean: _T, variance: _T) -> _T:
        log_scaled_sq_residual = - 0.5 * ((response - mode_mean).square() / variance).sum(-1)
        det_term = - 0.5 * variance.log().sum(-1)
        pi_term = - 0.5 * variance.shape[-1] * math.log(2 * math.pi)
        llh = log_scaled_sq_residual + pi_term + det_term
        return llh

    @abstractmethod
    def get_mixture_model_means_postcue(self, report_features: _T, probe_features: _T, cued_indices: _T):
        # batch_size = report_features.shape[0]
        # assert tuple(cued_indices.shape) == (batch_size, )
        # spline_means = self.get_mixture_model_spline_means(report_features)

        # cued_probe_features = probe_features[torch.arange(batch_size),cued_indices]
        # cued_gating_matrix = self.generate_gating_matrix(cued_probe_features, 'cued')
        # cued_mean_components = torch.einsum('bnk,bnrk->bnr', spline_means, cued_gating_matrix)
        # cued_mean_component = cued_mean_components.sum(1)

        # set_size = probe_features.shape[1]
        # uncued_indices = torch.tensor([[i for i in range(set_size) if i != ci] for ci in cued_indices])
        # uncued_probe_features = probe_features[torch.arange(batch_size),uncued_indices]
        # uncued_gating_matrix = self.generate_gating_matrix(uncued_probe_features, 'uncued')
        # uncued_mean_components = torch.einsum('bnk,bnrk->bnr', spline_means, uncued_gating_matrix)
        # uncued_mean_component = uncued_mean_components.sum(1)

        import pdb; pdb.set_trace()

        # return cued_mean_component + uncued_mean_component
        raise NotImplementedError

    @abstractmethod
    def generate_gating_matrix(self, probe_features: _T, gating_type: str) -> _T:
        """
        probe_features can be actual probe features, or the slot that was cued
            The latter is the Alleman case after cuing, i.e. binary cued or not for that trial
            The former is worked on in child classes, see below!

        outputs of shape [B, T, N, R, K]

        gating_type can be 'precued', 'cued', 'uncued'
        """



class CuedIndexDependentReportFeatureProjector(AllemanStyleRoleReportFeatureProjector):
    """
    Here, we just do what Alleman did, and learn Wu, Wl, Wlt, Wud, Wut, Wld
        as seperate dim_R x dim_K matrices, for each timestep
    """
    def __init__(self, dim_K: int, dim_R: int, precue_duration: int, postcue_duration: int, same_gating_across_time: bool, shared_p_misbind_over_time: bool, main_layers_sizes=None) -> None:
        super().__init__(dim_K, dim_R, precue_duration, postcue_duration, shared_p_misbind_over_time, main_layers_sizes)

        self.same_gating_across_time = same_gating_across_time  # Move to super init

        self.gating = nn.ModuleDict()
        gating_matrix_size = dim_K * dim_R if same_gating_across_time else dim_K * dim_R * precue_duration
        self.gating['precued_matrix'] = nn.Embedding(num_embeddings=2, embedding_dim=gating_matrix_size)
        gating_matrix_size = dim_K * dim_R if same_gating_across_time else dim_K * dim_R * postcue_duration
        for name in ['cued_matrix', 'uncued_matrix']:
            self.gating[name] = nn.Embedding(num_embeddings=2, embedding_dim=gating_matrix_size)

    def generate_gating_matrix(self, probe_features: _T, gating_type: str) -> _T:
        """
        Output is [B, T, N, R, K]

        If gating type is precued, then actual content of probe_features doesn't matter 
            - we just return output[:,:,0] = upper matrix and output[:,:,1] = lower matrix
        """
        assert len(probe_features.shape) == 1, f"Expected probe_features of shape [B], just indexes of which item was cued, got {probe_features.shape}"
        if gating_type == 'precued':
            
            probe_features = probe_features.int()
            upper_matrix = self.gating['precued_matrix'](torch.zeros_like(probe_features))
            lower_matrix = self.gating['precued_matrix'](torch.ones_like(probe_features))
            if self.same_gating_across_time:
                upper_matrix = upper_matrix.reshape(probe_features.shape[0], self.dim_R, self.dim_K).unsqueeze(-3).expand(probe_features.shape[0], self.precue_duration, self.dim_R, self.dim_K)
                lower_matrix = lower_matrix.reshape(probe_features.shape[0], self.dim_R, self.dim_K).unsqueeze(-3).expand(probe_features.shape[0], self.precue_duration, self.dim_R, self.dim_K)
            else:
                upper_matrix = upper_matrix.reshape(probe_features.shape[0], self.precue_duration, self.dim_R, self.dim_K)
                lower_matrix = lower_matrix.reshape(probe_features.shape[0], self.precue_duration, self.dim_R, self.dim_K)
            return torch.stack([upper_matrix, lower_matrix], dim = 2)

        if gating_type == 'precued':
            import pdb; pdb.set_trace()
            pass
    
    def get_mixture_model_means_postcue(self, report_features: _T, probe_features: _T, cued_indices: _T):
        """
        report_features if shaped [B, N, 2]
        in this case, probe_features is {0, 1}^B

        output is shaped [B, T, R]

        intermediary:
            spline_means: [B, N, K]

            upper_gating_matrices: [B, T, R, K]
                if cued_indices[b] == 0, then upper matrix was cued, so:
                    upper_gating_matrices[b] == reshaped(self.cued_matrix[0])
                    lower_gating_matrices[b] == reshaped(self.uncued_matrix[1])
                
                if cued_indices[b] == 1, then lower matrix was cued, so:
                    upper_gating_matrices[b] == reshaped(self.uncued_matrix[0])
                    lower_gating_matrices[b] == reshaped(self.cued_matrix[1])


            stacked_gating_matrices: [B, T, N, R, K] = stack(upper_gating_matrices, lower_gating_matrices)

            unmixed output [B, T, N, R] = report_features @ stacked_gating_matrices
            output [B, T, R]
        """
        
        assert (probe_features == cued_indices).all() or (probe_features == 1 - cued_indices).all()

        spline_means = self.get_mixture_model_spline_means(report_features)                 # [B, N, 2] -> [B, N, K]
        batch_size = spline_means.shape[0]


        upper_cued_mask = (cued_indices == 0).bool()
        lower_cued_mask = (cued_indices == 1).bool()
        assert torch.logical_xor(upper_cued_mask, lower_cued_mask).all()

        upper_gating_matrices = torch.zeros(batch_size, self.postcue_duration, self.dim_R, self.dim_K, device = report_features.device)
        lower_gating_matrices = torch.zeros(batch_size, self.postcue_duration, self.dim_R, self.dim_K, device = report_features.device)

        # XXX for the love of God fix this
        if self.same_gating_across_time:

            target_shape = self.dim_R, self.dim_K
            
            upper_gating_matrices[upper_cued_mask] = self.gating['cued_matrix'](torch.zeros(*upper_gating_matrices[upper_cued_mask].shape[:-2], device = report_features.device).int()).reshape(*upper_gating_matrices[upper_cued_mask].shape[:-2], *target_shape)
            upper_gating_matrices[lower_cued_mask] = self.gating['uncued_matrix'](torch.zeros(*upper_gating_matrices[lower_cued_mask].shape[:-2], device = report_features.device).int()).reshape(*upper_gating_matrices[lower_cued_mask].shape[:-2], *target_shape)

            lower_gating_matrices[lower_cued_mask] = self.gating['cued_matrix'](torch.ones(*lower_gating_matrices[lower_cued_mask].shape[:-2], device = report_features.device).int()).reshape(*lower_gating_matrices[lower_cued_mask].shape[:-2], *target_shape)
            lower_gating_matrices[upper_cued_mask] = self.gating['uncued_matrix'](torch.ones(*lower_gating_matrices[upper_cued_mask].shape[:-2], device = report_features.device).int()).reshape(*lower_gating_matrices[upper_cued_mask].shape[:-2], *target_shape)

        else:

            target_shape = self.postcue_duration, self.dim_R, self.dim_K            

            upper_gating_matrices[upper_cued_mask] = self.gating['cued_matrix'](torch.zeros(*upper_gating_matrices[upper_cued_mask].shape[:-3], device = report_features.device).int()).reshape(*upper_gating_matrices[upper_cued_mask].shape[:-3], *target_shape)
            upper_gating_matrices[lower_cued_mask] = self.gating['uncued_matrix'](torch.zeros(*upper_gating_matrices[lower_cued_mask].shape[:-3], device = report_features.device).int()).reshape(*upper_gating_matrices[lower_cued_mask].shape[:-3], *target_shape)

            lower_gating_matrices[lower_cued_mask] = self.gating['cued_matrix'](torch.ones(*lower_gating_matrices[lower_cued_mask].shape[:-3], device = report_features.device).int()).reshape(*lower_gating_matrices[lower_cued_mask].shape[:-3], *target_shape)
            lower_gating_matrices[upper_cued_mask] = self.gating['uncued_matrix'](torch.ones(*lower_gating_matrices[upper_cued_mask].shape[:-3], device = report_features.device).int()).reshape(*lower_gating_matrices[upper_cued_mask].shape[:-3], *target_shape)

        stacked_gating_matices = torch.stack([upper_gating_matrices, lower_gating_matrices], dim = 2)

        return torch.einsum('bnk,btnrk->btnr', spline_means, stacked_gating_matices).sum(-2)         # [B, T, R]




class ProbeFeatureDependentReportFeatureProjector(AllemanStyleRoleReportFeatureProjector):

    """
    Unlike Alleman et al., 2023, we do not have upper or lower item, the probe feature is circular for us too

    So we use:
        self.main_layers(report_features) -> like their f, sized K
        self.precued_layers(probe_features) -> Like their W, sized (N x K)
        self.cued_layers(probe_features) -> Like their Wt, sized (N x K)
        self.uncued_layers(probe_features) -> Like their Wd, sized (N x K)
    """

    def __init__(self, dim_K: int, dim_R: int, precue_duration: int, postcue_duration: int, same_gating_across_time: bool, shared_p_misbind_over_time: bool, main_layers_sizes=None, gate_layers_sizes=None) -> None:
        super().__init__(dim_K, dim_R, precue_duration, postcue_duration, shared_p_misbind_over_time, main_layers_sizes)
        
        self.same_gating_across_time = same_gating_across_time  # Move to super init

        if gate_layers_sizes is None:
            gate_layers_sizes = [dim_K * dim_R, dim_K * dim_R]

        gating_matrix_size = dim_K * dim_R if same_gating_across_time else dim_K * dim_R * precue_duration

        self.gating = nn.ModuleDict()
        for name in ['precued_layers', 'cued_layers', 'uncued_layers']:
            gate_layers = [nn.Linear(2, gate_layers_sizes[0]), nn.Sigmoid()]
            for h_in, h_out in zip(gate_layers_sizes[:-1], gate_layers_sizes[1:]):
                gate_layers.extend([nn.Linear(h_in, h_out), nn.Sigmoid()])
            gate_layers.extend([nn.Linear(gate_layers_sizes[-1], gating_matrix_size)])
            self.gating[name] = nn.Sequential(*gate_layers)

    def generate_gating_matrix(self, probe_features: _T, gating_type: str) -> _T:
        """
        Output is [B, T, N, R, K]

        if gating_type == 'precued': output[b,t,n] gives W(probe_feature[b,n]; t)
        """

        assert len(probe_features.shape) == 3, f"Expected probe_features of shape [B, N, Dr], got {probe_features.shape}"
        flat_matrices = self.gating[gating_type + '_layers'](probe_features)      # [B, N, TRK or RK]

        if self.same_gating_across_time:
            matrices = flat_matrices.reshape(*flat_matrices.shape[:2], self.dim_R, self.dim_K)
            matrices = matrices.unsqueeze(-4).expand(probe_features.shape[0], self.precue_duration, probe_features.shape[1], self.dim_R, self.dim_K)
            return matrices

        else:
            matrices = flat_matrices.reshape(*flat_matrices.shape[:2], self.precue_duration, self.dim_R, self.dim_K)
            matrices = matrices.transpose(-3, -4)
            return matrices
    
    def get_mixture_model_means_postcue(self, report_features: _T, probe_features: _T, cued_indices: _T):
        raise NotImplementedError


