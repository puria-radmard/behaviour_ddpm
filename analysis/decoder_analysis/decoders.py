import torch
from torch import nn
from torch import Tensor as _T



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





class AllemanStyleRoleReportFeatureProjectors(nn.Module):
    """
    Some form of stimulus_n -> r_n, allowing us to build a mixture model

    In Alleman et al. 2023, this is of the form:
        r(u, l) = Wu f(u) + Wl f(l) + η
    before cuing where:
        u, l are the upper and lower report values
        f is shared spline function (R^2 -> R^K)
        W. are projectors for the upper and lower functions,
        and η is noise

    We replace the 
    
    After cuing, we have:
        r(u,l,c) =  Wlt f(l) + Wud f(u)+η if c=0
                    Wut f(u) + Wld f(l)+η if c=1
    where:
        c = 0 means the lower colour is cued
        and 
    """

    def __init__(self, dim_K: int, main_layers_sizes = None) -> None:
        super().__init__()
        if main_layers_sizes is None:
            main_layers_sizes = [dim_K, dim_K, dim_K]



class ReportFeatureProjectors(nn.Module):

    """
    Unlike Alleman et al., 2023, we do not have upper or lower item, the probe feature is circular for us too

    So we use:
        self.main_layers(report_features) -> like their f, sized K
        self.precued_layers(probe_features) -> Like their W, sized (N x K)
        self.cued_layers(probe_features) -> Like their Wt, sized (N x K)
        self.uncued_layers(probe_features) -> Like their Wd, sized (N x K)
    """

    def __init__(self, dim_K, dim_R, main_layers_sizes = None, gate_layers_sizes = None) -> None:
        super().__init__()


        if gate_layers_sizes is None:
            gate_layers_sizes = [dim_K * dim_R]

        self.dim_K = dim_K
        self.dim_R = dim_R
        
        main_layers = [nn.Linear(2, main_layers_sizes[0]), nn.Sigmoid()]
        for h_in, h_out in zip(main_layers_sizes[:-1], main_layers_sizes[1:]):
            main_layers.extend([nn.Linear(h_in, h_out), nn.Sigmoid()])
        main_layers.extend([nn.Linear(main_layers_sizes[-1], dim_K)])
        self.main_layers = nn.Sequential(*main_layers)

        self.gating = nn.ModuleDict()
        for name in ['precued_layers', 'cued_layers', 'uncued_layers']:
            gate_layers = [nn.Linear(2, gate_layers_sizes[0]), nn.Sigmoid()]
            for h_in, h_out in zip(gate_layers_sizes[:-1], gate_layers_sizes[1:]):
                gate_layers.extend([nn.Linear(h_in, h_out), nn.Sigmoid()])
            gate_layers.extend([nn.Linear(gate_layers_sizes[-1], dim_R * dim_K)])
            self.gating[name] = nn.Sequential(*gate_layers)

    def get_mixture_model_spline_means(self, report_features: _T) -> _T:
        assert len(report_features.shape) == 3, f"Expected report_features of shape [B, N, Dr], got {report_features.shape}"
        return self.main_layers(report_features)

    def generate_gating_matrix(self, probe_features: _T, gating_type: str) -> _T:
        assert len(probe_features.shape) == 3, f"Expected probe_features of shape [B, N, Dr], got {probe_features.shape}"
        flat_matrix = self.gating[gating_type + '_layers'](probe_features)
        return flat_matrix.reshape(*flat_matrix.shape[:-1], self.dim_R, self.dim_K)
    
    def get_mixture_model_means_precue(self, report_features: _T, probe_features: _T):
        spline_means = self.get_mixture_model_spline_means(report_features)
        gating_matrix = self.generate_gating_matrix(probe_features, 'precued')
        return torch.einsum('bnk,bnrk->bnr', spline_means, gating_matrix).sum(1)
    
    def get_mixture_model_means_postcue(self, report_features: _T, probe_features: _T, cued_indices: _T):
        batch_size = report_features.shape[0]
        assert tuple(cued_indices.shape) == (batch_size, )
        spline_means = self.get_mixture_model_spline_means(report_features)

        cued_probe_features = probe_features[torch.arange(batch_size),cued_indices]
        cued_gating_matrix = self.generate_gating_matrix(cued_probe_features, 'cued')
        cued_mean_components = torch.einsum('bnk,bnrk->bnr', spline_means, cued_gating_matrix)
        cued_mean_component = cued_mean_components.sum(1)

        set_size = probe_features.shape[1]
        uncued_indices = torch.tensor([[i for i in range(set_size) if i != ci] for ci in cued_indices])
        uncued_probe_features = probe_features[torch.arange(batch_size),uncued_indices]
        uncued_gating_matrix = self.generate_gating_matrix(uncued_probe_features, 'uncued')
        uncued_mean_components = torch.einsum('bnk,bnrk->bnr', spline_means, uncued_gating_matrix)
        uncued_mean_component = uncued_mean_components.sum(1)

        return cued_mean_component + uncued_mean_component

