from dynamic_observer.model import *

from torch import Tensor as _T

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from matplotlib.pyplot import Axes


class PalimpsestRepresentation:
    """
    TODO: individual tuning scales
    """

    def __init__(self, probe_num_tc: int, report_num_tc: int, div_norm: bool, tuning_scale: float, resp_peak: float) -> None:
        self.probe_centers = torch.linspace(-torch.pi, +torch.pi, probe_num_tc + 1)[:-1]
        self.report_centers = torch.linspace(-torch.pi, +torch.pi, report_num_tc + 1)[:-1]
        self.probe_tuning_scales = torch.ones_like(self.probe_centers) * tuning_scale
        self.div_norm = div_norm
        self.report_tuning_scales = torch.ones_like(self.report_centers) * tuning_scale
        self.total_size = probe_num_tc * report_num_tc
        self.probe_num_tc = probe_num_tc
        self.report_num_tc = report_num_tc
        self.resp_peak = resp_peak

    @staticmethod
    def generate_responses(features: _T, centers: _T, scales: _T, peak: float, rescale: float) ->_T:
        """
        r_i(a) = exp(cos(a - a_i) * scale) / exp(scale)
        """
        centers = centers[*[None]*len(features.shape)]
        features = features.unsqueeze(-1)
        scaled_diffs = rectify_angles(centers - features).cos() * scales * rescale
        return scaled_diffs.exp() / (rescale * scales).exp() * (peak * rescale)
    
    def all_mean_responses(self, probe: _T, report: _T) -> Dict[str, _T]:
        """
        both of shape [..., N], where N is number of items
        """
        assert probe.shape == report.shape
        probe_repr = self.generate_responses(probe, self.probe_centers, self.probe_tuning_scales, self.resp_peak, 1.0)        # [..., N, probe size]
        report_repr = self.generate_responses(report, self.report_centers, self.report_tuning_scales, self.resp_peak, 1.0)
        probe_repr = probe_repr.unsqueeze(-2)
        report_repr = report_repr.unsqueeze(-1)
        joint_repr = (probe_repr * report_repr)  # [..., N, probe size, report size]
        if self.div_norm:
            joint_repr = joint_repr.mean(-3)  # [..., probe size, report size]
        else:
            joint_repr = joint_repr.sum(-3)
        joint_resp = joint_repr.reshape(*joint_repr.shape[:-2], -1)  # [..., total size]
        probe_resp = joint_repr.mean(-2)
        report_resp = joint_repr.mean(-1)
        return {
            "joint_resp": joint_resp,
            "probe_resp": probe_resp,
            "report_resp": report_resp,
        }

    def single_mean_response(self, feature_values: _T, feature_name: str, rescale: float):
        if feature_name == 'probe':
            feature_repr = self.generate_responses(feature_values, self.probe_centers, self.probe_tuning_scales, self.resp_peak, rescale)        # [..., N, probe size]
        elif feature_name == 'report':
            feature_repr = self.generate_responses(feature_values, self.report_centers, self.report_tuning_scales, self.resp_peak, rescale)        # [..., N, report size]
        set_size = feature_repr.shape[-2]
        if self.div_norm:
            feature_repr = feature_repr.mean(-2)  # [..., probe size]
        else:
            feature_repr = feature_repr.sum(-2)
        return feature_repr * feature_repr.mean() / set_size

    def generate_diffusion_conditioning(self, probe_values: _T, num_timesteps: int, rescale: float) -> Dict[str, _T]:
        """
        probe_values of shape [..., N]

        returns:
            A of shapes [num_timesteps, ..., probe population size, total population size]
            y of shape [num_timesteps, ..., probe population size]
            obs_covar of shape [num_timesteps, ..., probe population size, probe population size]
        """
        y = self.single_mean_response(feature_values = probe_values, feature_name = 'probe', rescale = rescale)
        
        obs_covar = 0.006 * torch.eye(self.probe_num_tc)[*[None]*(len(probe_values.shape)-1)].repeat(*probe_values.shape[:-1], 1, 1)

        A_squared_canvas = torch.zeros(*probe_values.shape[:-1], self.probe_num_tc, self.report_num_tc, self.probe_num_tc)
        for i in range(self.probe_num_tc):
            A_squared_canvas[...,i,:,i] = 1 / self.report_num_tc
        A = A_squared_canvas.reshape(*A_squared_canvas.shape[:-3], self.probe_num_tc, self.probe_num_tc * self.report_num_tc)
        
        ret = {'A': A, 'y': y, 'obs_covar': obs_covar}
        return {k: v[None].repeat_interleave(num_timesteps, 0) for k, v in ret.items()}

    def display_population_response(self, response: _T, axes: Axes, **kwargs):
        assert tuple(response.shape) == (self.total_size, ), response.shape
        reshaped_response = response.cpu().numpy().reshape(self.report_num_tc, self.probe_num_tc)
        return axes.imshow(reshaped_response, extent=(-torch.pi, torch.pi, torch.pi, -torch.pi), **kwargs)



class CircularDDM:

    def __init__(self, duration: float, discretiser: DynamicsDiscretiser, noise_mag: float) -> None:
        self.duration = duration
        self.discretiser = discretiser
        self.noise_mag = noise_mag
    
    def run_ddm(self, stimulus: _T, angles: _T) -> _T:
        """
        Expecting stimulus of shape [batch, num trials, num steps, num angles], and angles of shape [angles]

        TODO: allow time to warp!
        """
        batch, num_trials, num_steps, num_angles = stimulus.shape
        assert (num_angles,) == angles.shape

        dt = torch.tensor(self.duration / num_steps)

        angles_x, angles_y = angles.cos(), angles.sin() # both [num angles]
        resultants = (stimulus.unsqueeze(-1) * torch.stack([angles_x, angles_y], -1)[None, None, None]).mean(-2)    # [batch, num trials, num steps, 2]

        scaled_noisy_resultants = (dt * resultants) + (dt.sqrt() * self.noise_mag * torch.randn_like(resultants))    # [batch, num trials, num steps, 2]

        trajectory = scaled_noisy_resultants.cumsum(-2)  # [batch, num trials, 2]

        return trajectory



def run_custom_dynamics(
    start_samples: _T, smdm: ContinuousTimeScoreMatchingDiffusionModel, palimp: PalimpsestRepresentation,
    stimuli: Tuple[_T, ...], cued_probe_values: _T, num_steps: int
) -> _T:
    """
    start_samples of shape [B, D]

    cued_probe_values of shape [B]
    """
    start_samples = start_samples.unsqueeze(-2) # [B, 1, D]

    time: _T = torch.linspace(smdm.noise_schedule.duration, 0.0, num_steps+1)[1:]  # [num_steps]
    time = time.reshape(1, -1, 1)   # [1, num steps, 1]

    delta_t = - torch.diff(time, dim = len(time.shape)-2) # reverse time! [...1, num_steps - 1, 1]

    num_extra_steps = delta_t.shape[-2]

    beta = smdm.noise_schedule.beta(time)

    trajectory = [start_samples]

    score_approximator_dispatcher = smdm.score_approximator.prepare_dispatcher(
        stimuli = stimuli, t = time[0, ..., :-1, 0]        # Not sure why but [1:] doesn't work here...!
    )

    fig, axes = plt.subplots(1)

    current_cued_probe_values = cued_probe_values

    for t_tilde_idx in tqdm(range(num_extra_steps)):

        beta_k = beta[..., [t_tilde_idx], :]    # Not sure why but t_tilde_idx + 1 doesn't work here...!
        dt = delta_t[..., [t_tilde_idx], :]
        x_k = trajectory[-1]

        score_approx = score_approximator_dispatcher.approximate_score(x_t = x_k, t_tilde_idx = t_tilde_idx)  # [B, 1, D]

        if True:
            cue_dimension_activation = x_k.reshape(x_k.shape[0], palimp.report_num_tc, palimp.probe_num_tc).mean(-2)
            normed_cue_dimension_activation = (cue_dimension_activation / 2.0).softmax(-1)
            diffs = rectify_angles(cued_probe_values.unsqueeze(-1) - palimp.probe_centers.unsqueeze(0))
            force_on_cue_repr = (diffs * normed_cue_dimension_activation).sum(-1)

            current_cued_probe_values = rectify_angles(current_cued_probe_values + (0.25 * torch.randn_like(cued_probe_values)))
            current_cued_probe_values = rectify_angles(current_cued_probe_values + (0.25 * force_on_cue_repr))

            axes.plot(palimp.probe_centers, cue_dimension_activation[0], color = 'blue', alpha = (t_tilde_idx + 10) / (num_extra_steps + 10))

            diffusion_conditioning_info = palimp.generate_diffusion_conditioning(current_cued_probe_values.unsqueeze(1), num_timesteps=1, rescale = 1.5)
            
            A = diffusion_conditioning_info['A'][0] # one timestep
            obs = diffusion_conditioning_info['y'][0]   # one timestep
            obs_covar = diffusion_conditioning_info['obs_covar'][0] # one timestep
            obs_covar_inv = torch.inverse(obs_covar)
            
            assert x_k.shape[-2] == 1

            conditioned_residual = obs - torch.einsum('...ij,...j->...i', A, x_k.squeeze(-2))
            conditioned_score = torch.einsum('...ij,...ik,...k->...j', A, obs_covar_inv, conditioned_residual)
            
            score_approx = score_approx + conditioned_score.unsqueeze(-2)

    
        drift = - (0.5 * beta_k * x_k) - (beta_k * score_approx)    # f(x, t) - g(t)^2 * s(x, t)
        diffusion = beta_k.sqrt()            # g(t) dW_t
        next_step = smdm.discretiser.step(x_k, - drift, diffusion, dt)    # Reverse time!
        trajectory.append(next_step)

    fig.savefig('asdf2')

    return torch.concat(trajectory, -2)





if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    #### SETUP
    num_selected_timesteps = 10
    steps_between_displays = 10
    batch_size = 4
    num_reverse_dynamics_steps = num_selected_timesteps * steps_between_displays + 1

    num_ddm_trials = 100

    palimpsest = PalimpsestRepresentation(probe_num_tc = 16, report_num_tc = 16, div_norm = False, tuning_scale = 2.0, resp_peak = 3.0)

    probes = torch.tensor([-torch.pi * 2 / 3, torch.pi * 2 / 3, 0.0])
    reports = torch.tensor([-3.0, 1.6, 0.0])
    cued_idx = 2

    noise_schedule = LinearIncreaseNoiseSchedule(0.08, 0.15, duration = 15)

    score_func = TrueScore(noise_schedule = noise_schedule)

    discretiser = EulerDiscretiser()
    
    diffmodel = ContinuousTimeScoreMatchingDiffusionModel(
        sample_dim=palimpsest.total_size, noise_schedule=noise_schedule, 
        score_approximator=score_func, discretiser=discretiser
    )
    print('DDM HAS A DIFFERENT DURIATION TO DM')
    ddm = CircularDDM(duration = 1.0, discretiser=discretiser, noise_mag = 0.25)

    all_mean_responses = palimpsest.all_mean_responses(probes, reports)
    diffusion_conditioning_info = palimpsest.generate_diffusion_conditioning(probes[[cued_idx]], num_timesteps=num_reverse_dynamics_steps - 1, rescale = 1.5)
    projected_mean_resposes = diffusion_conditioning_info['A'][0] @ all_mean_responses['joint_resp']

    target_m0 = all_mean_responses['joint_resp'][None,None].repeat(num_reverse_dynamics_steps - 1, batch_size, 1)
    target_S0 = torch.eye(palimpsest.total_size)[None,None].repeat(num_reverse_dynamics_steps - 1, batch_size, 1, 1)
    target_S0 = target_S0 * target_m0.unsqueeze(-1) / 10
    palimp_stimulus = (target_m0, target_S0)

    ddm_driving_mult = 5.0
    ##########


    #### RUN DIFFUSION MODEL
    base_samples = torch.randn(batch_size, palimpsest.total_size)
    all_reverse_trajectories = diffmodel.run_unconditioned_reverse_dynamics(base_samples, palimp_stimulus, num_reverse_dynamics_steps)

    base_samples = torch.randn(batch_size, palimpsest.total_size)
    all_conditioned_reverse_trajectories = diffmodel.run_conditioned_reverse_dynamics(
        base_samples, palimp_stimulus, num_reverse_dynamics_steps, 
        diffusion_conditioning_info['y'], diffusion_conditioning_info['A'], diffusion_conditioning_info['obs_covar']
    )

    base_samples = torch.randn(batch_size, palimpsest.total_size)
    all_custom_reverse_trajectories = run_custom_dynamics(
        start_samples=base_samples, smdm=diffmodel, palimp=palimpsest,
        stimuli = palimp_stimulus, cued_probe_values=probes[[cued_idx]].repeat(4),
        num_steps = num_reverse_dynamics_steps
    )
    ##########



    #### RUN DDM
    uncond_ddm_driver = all_reverse_trajectories[0].reshape(num_reverse_dynamics_steps, palimpsest.report_num_tc, palimpsest.probe_num_tc).max(-1).values
    unconditional_ddm_dynamics = ddm.run_ddm(ddm_driving_mult * uncond_ddm_driver[None,None].repeat(1, num_ddm_trials, 1, 1), palimpsest.report_centers)

    cond_ddm_driver = all_conditioned_reverse_trajectories[0].reshape(num_reverse_dynamics_steps, palimpsest.report_num_tc, palimpsest.probe_num_tc).max(-1).values
    conditional_ddm_dynamics = ddm.run_ddm(ddm_driving_mult * cond_ddm_driver[None,None].repeat(1, num_ddm_trials, 1, 1), palimpsest.report_centers)

    custom_ddm_driver = all_custom_reverse_trajectories[0].reshape(num_reverse_dynamics_steps, palimpsest.report_num_tc, palimpsest.probe_num_tc).max(-1).values
    custom_ddm_dynamics = ddm.run_ddm(ddm_driving_mult * custom_ddm_driver[None,None].repeat(1, num_ddm_trials, 1, 1), palimpsest.report_centers)
    ##########




    #### PLOT
    fig, axes = plt.subplots(1, 2, figsize = (15, 10))
    ims = palimpsest.display_population_response(all_mean_responses['joint_resp'], axes[0])
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ims, cax=cax, orientation='vertical')

    axes[1].plot(palimpsest.probe_centers.cpu().numpy(), all_mean_responses['probe_resp'].cpu().numpy(), label = 'probe from joint')
    axes[1].plot(palimpsest.probe_centers.cpu().numpy(), diffusion_conditioning_info['y'][0].cpu().numpy(), label = 'probe alone')
    axes[1].plot(palimpsest.probe_centers.cpu().numpy(), projected_mean_resposes.cpu().numpy(), label = 'projected')
    axes[1].legend()

    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/z_palimpsest_sampling/palimpsest_mean_resp.png')

    fig, axes = plt.subplots(1, figsize = (5, 5))
    scaling_factor_time, scaling_factor = noise_schedule.summarise_noising_factor(100)
    axes.plot(scaling_factor_time.cpu().numpy(), scaling_factor.cpu().numpy())
    axes.set_ylim(0)

    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/z_palimpsest_sampling/palimpsest_schedule.png')

    fig, axes = plt.subplots(3, num_selected_timesteps + 2, figsize = ((2 + num_selected_timesteps) * 5, 3 * 5))

    selected_timesteps = range(steps_between_displays, num_reverse_dynamics_steps+1, steps_between_displays)

    for i_a, i_t in enumerate(selected_timesteps):
        axes[0,i_a].set_title(f'reverse step: {i_t+1} of {num_reverse_dynamics_steps}')
        im_unc = palimpsest.display_population_response(all_reverse_trajectories[0,i_t], axes[0,i_a])
        im_cond = palimpsest.display_population_response(all_conditioned_reverse_trajectories[0,i_t], axes[1,i_a])
        im_cust = palimpsest.display_population_response(all_custom_reverse_trajectories[0,i_t], axes[2,i_a])
        
        axes[0,-2].plot(uncond_ddm_driver[i_t], - palimpsest.report_centers, label = i_t + 1)
        axes[1,-2].plot(cond_ddm_driver[i_t], - palimpsest.report_centers, label = i_t + 1)
        axes[2,-2].plot(custom_ddm_driver[i_t], - palimpsest.report_centers, label = i_t + 1)


        divider = make_axes_locatable(axes[0,i_a])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_unc, cax=cax, orientation='vertical')

        divider = make_axes_locatable(axes[1,i_a])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_cond, cax=cax, orientation='vertical')

        divider = make_axes_locatable(axes[2,i_a])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_cust, cax=cax, orientation='vertical')

    axes[0,-1].legend()
    axes[1,-1].legend()


    for i_tr in range(num_ddm_trials):
        axes[0,-1].plot(* unconditional_ddm_dynamics[:,i_tr].T, color = 'gray', alpha = 0.1)
        axes[1,-1].plot(* conditional_ddm_dynamics[:,i_tr].T, color = 'gray', alpha = 0.1)
        axes[2,-1].plot(* custom_ddm_dynamics[:,i_tr].T, color = 'gray', alpha = 0.1)

    axes[0,-1].set_aspect(1.0)
    axes[1,-1].set_aspect(1.0)
    axes[2,-1].set_aspect(1.0)

    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/z_palimpsest_sampling/palimpsest_samples.png')
    ##########


