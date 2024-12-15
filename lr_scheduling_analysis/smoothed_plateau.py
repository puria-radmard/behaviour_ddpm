import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

results_path = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_unrolling_26_11_24/unrolling_N15_s0.1_tp3Eo_0/epoch_log_train_T2.csv'
fig_output_path = '/homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/lr_scheduling_analysis/cutoff.png'

results = pd.read_csv(results_path, sep = '\t')

avg_recon_loss = results.avg_recon_loss
avg_kl_loss_1 = results.avg_kl_loss_1

smoothing_lengths = [1000]
patiences = [50]# np.linspace(50, 150, 5).astype(int)

fig, axes = plt.subplots(2)

for sl in tqdm(smoothing_lengths):
    
    smoothed_avg_kl_loss_1 = np.convolve(avg_kl_loss_1, np.ones(sl)/sl, mode='valid')
    col = axes[0].plot(smoothed_avg_kl_loss_1, label = sl)[0].get_color()

    axes[1].plot(smoothed_avg_kl_loss_1[600_000:])

    is_decreasing = smoothed_avg_kl_loss_1[1:] < smoothed_avg_kl_loss_1[:-1]
    import pdb; pdb.set_trace()

    for pat in patiences:
        for i in range(len(is_decreasing)):
            if is_decreasing[i-pat:i].all():
                axes[0].scatter([i], smoothed_avg_kl_loss_1[[i]], color = col, label = pat)

plt.legend()

plt.savefig(fig_output_path)

import pdb; pdb.set_trace()
