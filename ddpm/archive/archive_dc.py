raise Exception('reintroduce soon!')




class DoublyConditionedResidualModel(ResidualModel):
    """
    Same as base model but input (estimates of) the 'final' mean as well
    Nothing much changes if you just double the state space size
    """
    def __init__(self, state_space_size: int, recurrence_hidden_layers: List[int], input_size: int, time_embedding_size: int) -> None:
        super().__init__(2 * state_space_size, recurrence_hidden_layers, input_size, time_embedding_size)
        self.layers.append(nn.Softplus())
        self.layers.append(nn.Linear(2 * state_space_size + self.input_size + self.time_embedding_size, state_space_size))
        self.state_space_size = state_space_size

    def forward(self, x: _T, final_mean: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x and final_mean of shape [..., T, state_space_size]
        t_embeddings_schedule of shape [T, time_emb_size]
        input_vector of shape [...], passed to all
        """
        x_and_final_mean = torch.concat([x, final_mean], -1)
        return super().forward(x_and_final_mean, t_embeddings_schedule, input_vector)
    




class DoublyConditionedDDPMReverseProcess(DDPMReverseProcess):

    def __init__(self, hidden_size: int, residual_model: ResidualModel, input_model: InputModelBlock, sigma2xt_schedule: _T, time_embedding_size: int, sample_space_size: int = 2, euler_alpha: float = 0.1) -> None:
        assert isinstance(residual_model, DoublyConditionedResidualModel)
        super().__init__(hidden_size, residual_model, input_model, sigma2xt_schedule, time_embedding_size, sample_space_size, euler_alpha)
        
        # Generation again
        self.final_mean_scaler_schedule = (
            ((1.0 - self.sigma2xt_schedule - self.a_t_schedule.square()) / (self.root_b_t_schedule.square() * (1 - self.sigma2xt_schedule)))
            - (1.0 / (1 - self.sigma2xt_schedule).sqrt())
            - ((self.sigma2xt_schedule * self.a_t_schedule * (1 - self.a_t_schedule)) / ((1 - self.a_t_schedule.square()) * (1 - self.sigma2xt_schedule).sqrt()))
        )

    def to(self, *args, **kwargs):
        self.final_mean_scaler_schedule = self.final_mean_scaler_schedule.to(*args, **kwargs)
        return super(DoublyConditionedDDPMReverseProcess, self).to(*args, **kwargs)

    def noise(self, y_samples: _T, final_mean: _T) -> Dict[str, _T]:
        """
        y_samples and final_mean of shape [..., dim y] --> same mean for all timesteps during training!

        Both outputs of shape [..., T, dim x]
        """
        extra_dims = len(y_samples.shape[:-1])
        
        # [..., 1, dim x]
        x_0 = (y_samples @ self.behaviour_projection_matrix.T).unsqueeze(-2)
        x_0 = x_0.repeat(*[1]*extra_dims, self.T, 1)
        epsilon = torch.randn_like(x_0)

        # [..., T, dim x]
        reshaped_a_t_schedule = self.a_t_schedule[*[None]*extra_dims].unsqueeze(-1)
        x_t_means = (reshaped_a_t_schedule * x_0) + (1.0 - reshaped_a_t_schedule) * final_mean.unsqueeze(-2)
        x_t = x_t_means + self.root_b_t_schedule[*[None]*extra_dims].unsqueeze(-1) * epsilon

        return {
            'x_t': x_t,
            'epsilon': epsilon
        }

    def residual(self, x_samples: _T, final_mean: _T, network_input: _T, epsilon: Optional[_T] = None) -> Dict[str, _T]:
        """
        final_mean of shape [..., T, dim y] ---> T dimension allows for noisy estimates later on!
        """
        return super().residual(x_samples = x_samples, network_input = network_input, epsilon = epsilon, final_mean = final_mean)

    def generate_samples(self, network_input: _T, final_mean: _T, samples_shape: Optional[int] = None, base_samples: Optional[_T] = None, turn_off_noise: bool = False) -> Dict[str, _T]:
        """
        input_vector of shape [..., <shape Z>]
        final_mean of shape [..., T, dim y] ---> T dimension allows for noisy estimates later on!
        If provided, base_samples of shape [..., dim x]
        Otherwise, give samples_shape = [...]
        
        final_mean of shape [..., T, dim y] ---> T dimension allows for noisy estimates later on!
            NB: this is in reverse order - see indexing below!

        TODO: so much of this could be inherited!
        """
        assert (samples_shape is None) != (base_samples is None)

        input_vector = self.input_model(network_input)

        if base_samples is None:
            base_samples = torch.randn(*samples_shape, self.hidden_size, device = self.sigma2xt_schedule.device) * self.base_std
        else:
            samples_shape = base_samples.shape[:-1]
            assert base_samples.shape[-1] == self.hidden_size

        sample_trajectory = []
        noises = torch.randn(*samples_shape, self.T, self.hidden_size, device = base_samples.device)
        if turn_off_noise:
            noises *= 0.0

        base_samples = base_samples.unsqueeze(-2)

        t_embeddings = self.time_embeddings(self.t_schedule)

        for t_idx in range(1, self.T):

            t_embedding = t_embeddings[-t_idx][None]
            noise = noises[...,[-t_idx],:]
            base_samples_scaler = self.base_samples_scaler_schedule[-t_idx]
            noise_scaler = self.noise_scaler_schedule[-t_idx]
            residual_scaler = self.residual_scaler_schedule[-t_idx]
            final_mean_scaler = self.final_mean_scaler_schedule[-t_idx]
            final_mean_estimate = final_mean[...,[t_idx-1],:]
            
            new_residual = self.residual_model(base_samples, final_mean_estimate, t_embedding, input_vector) # [..., 1, dim x]
            scaled_residual = residual_scaler * new_residual
            scaled_mean = base_samples / base_samples_scaler
            scaled_noise = noise * noise_scaler

            scaled_final_mean_estimate = final_mean_scaler * final_mean_estimate

            base_samples = scaled_mean - scaled_residual + scaled_final_mean_estimate + scaled_noise

            sample_trajectory.append(base_samples.cpu().detach())
        
        sample_trajectory = torch.concat(sample_trajectory, -2) # [..., T, dim x]
        behaviour_samples = base_samples.squeeze(-2) @ self.linking_matrix.T

        return {
            'sample_trajectory': sample_trajectory,
            'behaviour_samples': behaviour_samples
        }
            
            

