import torch

import numpy as np

import copy

import matplotlib.cm as cmx
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from purias_utils.util.arguments_yaml import ConfigNamepace

import sys

from drl.util import make_model



args = ConfigNamepace.from_yaml_path(sys.argv[1])


device = "cuda" if torch.cuda.is_available() else "cpu"


num_batches = args.num_batches
batch_size = args.batch_size
task_time_embedding_size = args.task_time_embedding_size
wait_time = args.wait_time
time_between_cs_and_us = args.time_between_cs_and_us
time_after_us = args.time_after_us
show_freq = args.show_freq
start_freezing_batch = float(args.start_freezing_batch)
freezing_frequency = args.freezing_frequency
gamma = args.gamma
opt_steps_per_batch = args.opt_steps_per_batch
all_reward_distribution_configs = args.all_reward_distributions
#reward_distribution_name = args.reward_distribution_name
#reward_distribution_kwargs = args.reward_distribution_kwargs.dict
#reward_distribution_min = args.reward_distribution_min
#reward_distribution_max = args.reward_distribution_max
starting_sigma2 = args.starting_sigma2
ultimate_sigma2 = args.ultimate_sigma2
num_diffusion_timesteps = args.num_diffusion_timesteps
resume_path = args.resume_path
save_base = args.save_base





sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_diffusion_timesteps)
std_schedule = torch.sqrt(sigma2x_schedule)
diffusion_time_embedding_size = 16

magma = plt.get_cmap("magma")
cNorm = colors.Normalize(vmin=1, vmax=num_diffusion_timesteps)
diffusion_timesteps_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
diffusion_timesteps_colors_scalarMap.set_array([])


magma = plt.get_cmap("viridis")
cNorm = colors.Normalize(vmin=-wait_time, vmax=time_between_cs_and_us)
task_timesteps_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
task_timesteps_colors_scalarMap.set_array([])





ddpm_model = make_model(task_time_embedding_size, diffusion_time_embedding_size, sigma2x_schedule, time_between_cs_and_us, len(all_reward_distribution_configs), device)

if resume_path is not None:
    ddpm_model.load_state_dict(torch.load(resume_path))



# ddpm_model.load_state_dict(torch.load('/homes/pr450/repos/research_projects/sampling_ddpm/results_link_drl/test/run_0/state.mdl'))

lr = 1e-4
logging_freq = 100
optim = torch.optim.Adam(ddpm_model.parameters(), lr = lr)

all_stepwise_losses = np.zeros([num_batches, num_diffusion_timesteps])
all_mean_losses = np.zeros([num_batches])


