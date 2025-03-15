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
