import torch
from torch import nn
from torch import Tensor as _T


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

        if main_layers_sizes is None:
            main_layers_sizes = [dim_K, dim_K, dim_K]
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



import torch, sys, os

from analysis.decoder_analysis.decoders import *

from tqdm import tqdm
import numpy as np

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from purias_utils.util.arguments_yaml import ConfigNamepace

from scipy.stats import ttest_rel

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1])
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]


# run_name = 'run_b2_probe_cued_with_probe_flat_swap_fewer_variable_delay_0'
run_name = analysis_args.run_name

device = 'cuda'
_, task, ddpm_model, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_further_20250120/{run_name}/args.yaml', device)
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_further_20250120/{run_name}/state.mdl'))

ddpm_model.eval()

num_neurons = ddpm_model.sample_ambient_dim

if analysis_args.precue:
    fixed_delay_duration = max(task.task_variable_gen.prep_epoch_durations[1])
    decoded_timestep = -5    # 5 steps before cue is presented...
else:
    raise NotImplementedError

random_choice = analysis_args.random_choice


decoder = ReportFeatureProjectors(dim_K = 5, dim_R = num_neurons)
decoder.cuda()

optim = torch.optim.Adam(decoder.parameters(), lr = 0.1)

save_path = os.path.join(analysis_args.save_base, yaml_name)
try:
    os.mkdir(save_path)
except FileExistsError:
    print('OVERWRITING RESULTS')
    try:
        decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.mdl')))
    except (FileNotFoundError, RuntimeError):
        print('NO SAVED MODEL TO RESUME FROM')

analysis_args.write_to_yaml(os.path.join(save_path, "args.yaml"))

ctd_batch_size = analysis_args.ctd_batch_size
num_opt_steps = analysis_args.num_opt_steps

decoding_variable = 'report_features_cart'
context_variable = 'probe_features_cart'

display_model_idx = 0

all_positional_decoder_losses = np.zeros([num_opt_steps])
all_shuffled_positional_decoder_losses = np.zeros([num_opt_steps])

num_samples = 1

infinite_data_mode = True


for i in tqdm(range(num_opt_steps)):

    if infinite_data_mode or i == 0:

        trial_information = task.generate_trial_information(batch_size=ctd_batch_size, num_samples=num_samples)

        trial_information.task_variable_information['prep_epoch_durations'][1] = fixed_delay_duration

        with torch.no_grad():
            novel_samples_prep_dicts, novel_samples_dict = ddpm_model.generate_samples(
                # prep_network_inputs=[
                #     pni[[0]] for pni in trial_information.prep_network_inputs
                # ],
                # diffusion_network_inputs=[
                #     dni[[0]] for dni in trial_information.diffusion_network_inputs
                # ],
                prep_network_inputs = trial_information.prep_network_inputs,
                diffusion_network_inputs = trial_information.diffusion_network_inputs,
                prep_epoch_durations=trial_information.prep_epoch_durations,
                diffusion_epoch_durations=trial_information.diffusion_epoch_durations,
                samples_shape=[ctd_batch_size, num_samples],
                noise_scaler=1.0,
            )

        relevant_activity = novel_samples_prep_dicts[1]['preparatory_trajectory'].squeeze(1)[:,decoded_timestep,:]  # [batch dim (128)]

    report = trial_information.task_variable_information[decoding_variable].cuda().to(relevant_activity.dtype)
    probe = trial_information.task_variable_information[context_variable].cuda().to(relevant_activity.dtype)
    output = decoder.get_mixture_model_means_precue(report_features=report, probe_features=probe)
    

    with torch.no_grad():
        shuffled_probe = probe[torch.randperm(probe.shape[0])]
        shuffled_output = decoder.get_mixture_model_means_precue(report_features=report, probe_features=shuffled_probe)

        all_trials_shuffled_loss = (relevant_activity - shuffled_output).square().sum(-1).sqrt()  # XXX: shape!
        all_shuffled_positional_decoder_losses[i] = all_trials_shuffled_loss.mean().item()

    all_trials_loss = (relevant_activity - output).square().sum(-1).sqrt()  # batch
    loss = all_trials_loss.mean()  # XXX: shape!

    optim.zero_grad()
    loss.backward()
    optim.step()

    all_positional_decoder_losses[i] = loss.item()

    if i % 50 == 0:

        plt.close('all')


        ### PLOT TRAINING LOSSES
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))

        zoomed_positional_decoder_losses_line = all_positional_decoder_losses[i+1-50:i+1]
        zoomed_shuffled_positional_decoder_losses_line = all_shuffled_positional_decoder_losses[i+1-50:i+1]
        positional_decoder_losses_line = all_positional_decoder_losses[:i+1]
        shuffled_positional_decoder_losses_line = all_shuffled_positional_decoder_losses[:i+1]

        axes[0,0].plot(zoomed_positional_decoder_losses_line)
        axes[0,1].plot(zoomed_shuffled_positional_decoder_losses_line)
        axes[1,0].plot(positional_decoder_losses_line)
        axes[1,1].plot(shuffled_positional_decoder_losses_line)
        
        axes[0,0].set_title('zoomed_positional_decoder_losses_line')
        axes[0,1].set_title('zoomed_shuffled_positional_decoder_losses_line')
        axes[1,0].set_title('positional_decoder_losses_line')
        axes[1,1].set_title('shuffled_positional_decoder_losses_line')

        fig.savefig(os.path.join(save_path, 'loss'))

        
        ### HIST SHUFFLED VS REAL LOSSES!
        fig, axes = plt.subplots(1, figsize = (8, 8))
        axes.hist(all_trials_loss.detach().cpu().numpy(), 64, alpha = 0.3, label = 'real euclidean distances')
        axes.hist(all_trials_shuffled_loss.detach().cpu().numpy(), 64, alpha = 0.3, label = 'shuffled probes euclidean distances')
        axes.legend()
        fig.savefig(os.path.join(save_path, 'loss_vs_shuffled'))

        
        torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.mdl'))

