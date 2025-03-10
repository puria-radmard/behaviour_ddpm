from dynamic_observer.ct_scorematching import *

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
        
        obs_covar = 0.008 * torch.eye(self.probe_num_tc)[*[None]*(len(probe_values.shape)-1)].repeat(*probe_values.shape[:-1], 1, 1)

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

    def __init__(self, duration: float, discretiser: DynamicsDiscretiser) -> None:
        self.duration = duration
        self.discretiser = discretiser
    
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

        scaled_noisy_resultants = (dt * resultants) + (dt.sqrt() * torch.randn_like(resultants))    # [batch, num trials, num steps, 2]

        trajectory = scaled_noisy_resultants.cumsum(-2)  # [batch, num trials, 2]

        return trajectory




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    num_selected_timesteps = 10
    steps_between_displays = 10
    batch_size = 4
    num_reverse_dynamics_steps = num_selected_timesteps * steps_between_displays + 1

    num_ddm_trials = 100

    palimpsest = PalimpsestRepresentation(probe_num_tc = 15, report_num_tc = 15, div_norm = False, tuning_scale = 3.0, resp_peak = 2.5)

    probes = torch.tensor([1.0, -1.5, 0.0])
    reports = torch.tensor([-3.0, 2.0, 0.0])

    all_mean_responses = palimpsest.all_mean_responses(probes, reports)
    diffusion_conditioning_info = palimpsest.generate_diffusion_conditioning(probes[[2]], num_timesteps=num_reverse_dynamics_steps - 1, rescale = 2.0)
    projected_mean_resposes = diffusion_conditioning_info['A'][0] @ all_mean_responses['joint_resp']
    
    fig, axes = plt.subplots(1, 2, figsize = (15, 10))
    ims = palimpsest.display_population_response(all_mean_responses['joint_resp'], axes[0])
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ims, cax=cax, orientation='vertical')

    axes[1].plot(palimpsest.probe_centers.cpu().numpy(), all_mean_responses['probe_resp'].cpu().numpy(), label = 'probe from joint')
    axes[1].plot(palimpsest.probe_centers.cpu().numpy(), diffusion_conditioning_info['y'][0].cpu().numpy(), label = 'probe alone')
    axes[1].plot(palimpsest.probe_centers.cpu().numpy(), projected_mean_resposes.cpu().numpy(), label = 'projected')
    axes[1].legend()

    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/palimpsest_mean_resp.png')

    noise_schedule = LinearIncreaseNoiseSchedule(0.15, 0.15, duration = 50)

    fig, axes = plt.subplots(1, figsize = (5, 5))
    scaling_factor_time, scaling_factor = noise_schedule.summarise_noising_factor(100)
    axes.plot(scaling_factor_time.cpu().numpy(), scaling_factor.cpu().numpy())
    axes.set_ylim(0)

    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/palimpsest_schedule.png')


    score_func = TrueScore(noise_schedule = noise_schedule)

    discretiser = EulerDiscretiser()
    
    diffmodel = ContinuousTimeScoreMatchingDiffusionModel(
        sample_dim=palimpsest.total_size, noise_schedule=noise_schedule, 
        score_approximator=score_func, discretiser=discretiser
    )

    print('DDM HAS A DIFFERENT DURIATION TO DM')
    ddm = CircularDDM(duration = 1.0, discretiser=discretiser)

    target_m0 = all_mean_responses['joint_resp'][None,None].repeat(num_reverse_dynamics_steps - 1, batch_size, 1)
    target_S0 = torch.eye(palimpsest.total_size)[None,None].repeat(num_reverse_dynamics_steps - 1, batch_size, 1, 1)
    target_S0 = target_S0 * target_m0.unsqueeze(-1) / 10
    stimulus = (target_m0, target_S0)

    ddm_driving_mult = 0.0

    base_samples = torch.randn(batch_size, palimpsest.total_size)
    all_reverse_trajectories = diffmodel.run_unconditioned_reverse_dynamics(base_samples, stimulus, num_reverse_dynamics_steps)
    uncond_ddm_driver = all_reverse_trajectories[0].reshape(num_reverse_dynamics_steps, palimpsest.report_num_tc, palimpsest.probe_num_tc).max(-1).values
    unconditional_ddm_dynamics = ddm.run_ddm(ddm_driving_mult * uncond_ddm_driver[None,None].repeat(1, num_ddm_trials, 1, 1), palimpsest.report_centers)

    base_samples = torch.randn(batch_size, palimpsest.total_size)
    all_conditioned_reverse_trajectories = diffmodel.run_conditioned_reverse_dynamics(
        base_samples, stimulus, num_reverse_dynamics_steps, 
        diffusion_conditioning_info['y'], diffusion_conditioning_info['A'], diffusion_conditioning_info['obs_covar']
    )
    cond_ddm_driver = all_conditioned_reverse_trajectories[0].reshape(num_reverse_dynamics_steps, palimpsest.report_num_tc, palimpsest.probe_num_tc).max(-1).values
    conditional_ddm_dynamics = ddm.run_ddm(ddm_driving_mult * cond_ddm_driver[None,None].repeat(1, num_ddm_trials, 1, 1), palimpsest.report_centers)

    # y_noise = torch.randn_like(diffusion_conditioning_info['y']) * diffusion_conditioning_info['y']
    # noisy_y = y_noise + diffusion_conditioning_info['y']

    # base_samples = torch.randn(batch_size, palimpsest.total_size)
    # all_noisy_conditioned_reverse_trajectories = diffmodel.run_conditioned_reverse_dynamics(
    #     base_samples, stimulus, num_reverse_dynamics_steps, 
    #     noisy_y, diffusion_conditioning_info['A'], diffusion_conditioning_info['obs_covar']
    # )

    fig, axes = plt.subplots(2, num_selected_timesteps + 2, figsize = ((2 + num_selected_timesteps) * 5, 3 * 5))

    selected_timesteps = range(steps_between_displays, num_reverse_dynamics_steps+1, steps_between_displays)

    for i_a, i_t in enumerate(selected_timesteps):
        axes[0,i_a].set_title(f'reverse step: {i_t+1} of {num_reverse_dynamics_steps}')
        im_unc = palimpsest.display_population_response(all_reverse_trajectories[0,i_t], axes[0,i_a])
        im_cond = palimpsest.display_population_response(all_conditioned_reverse_trajectories[0,i_t], axes[1,i_a])
        axes[0,-2].plot(uncond_ddm_driver[i_t], palimpsest.report_centers, label = i_t + 1)
        axes[1,-2].plot(cond_ddm_driver[i_t], palimpsest.report_centers, label = i_t + 1)

    axes[0,-1].legend()
    axes[1,-1].legend()

    divider = make_axes_locatable(axes[0,i_a])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_unc, cax=cax, orientation='vertical')

    divider = make_axes_locatable(axes[1,i_a])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_cond, cax=cax, orientation='vertical')


    for i_tr in range(num_ddm_trials):
        axes[0,-1].plot(* unconditional_ddm_dynamics[:,i_tr].T, color = 'gray', alpha = 0.1)
        axes[1,-1].plot(* conditional_ddm_dynamics[:,i_tr].T, color = 'gray', alpha = 0.1)

    axes[0,-1].set_aspect(1.0)
    axes[1,-1].set_aspect(1.0)

    fig.savefig('/homes/pr450/repos/research_projects/sampling_ddpm/dynamic_observer/palimpsest_samples.png')
