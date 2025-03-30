import torch, sys, os

from analysis.decoders import *

from tqdm import tqdm
import numpy as np

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from purias_utils.util.arguments_yaml import ConfigNamepace

from scipy.stats import ttest_rel

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch, generate_model_and_task_from_args_path


analysis_args = ConfigNamepace.from_yaml_path(sys.argv[1])
yaml_name = sys.argv[1].split('/')[-1].split('.')[0]


# run_name = 'run_b2_probe_cued_with_probe_flat_swap_fewer_variable_delay_0'
run_name = analysis_args.run_name

device = 'cuda'
try:
    _, task, ddpm_model, _ = generate_model_and_task_from_args_path_multiepoch(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_curriculum_activity_reg_20250322/{run_name}/args.yaml', device)
    num_neurons = ddpm_model.sample_ambient_dim
except:
    _, task, ddpm_model, _ = generate_model_and_task_from_args_path(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_curriculum_activity_reg_20250322/{run_name}/args.yaml', device)
ddpm_model.load_state_dict(torch.load(f'/homes/pr450/repos/research_projects/sampling_ddpm/results_link_sampler/ddpm_curriculum_activity_reg_20250322/{run_name}/state.mdl'))

ddpm_model.eval()

