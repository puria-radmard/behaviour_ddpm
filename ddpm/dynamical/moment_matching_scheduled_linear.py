from ddpm.dynamical.moment_matching_scheduled_utils import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os, sys

from purias_utils.util.logging import configure_logging_paths


from ddpm.model.input import InputModelBlock
from ddpm.model.main.base.hvae import LinearSubspaceTeacherForcedHVAEReverseProcess
from ddpm.model.transition.vectoral import DynamicalSystemTransitionModel
from purias_utils.util.arguments_yaml import ConfigNamepace
from sklearn.decomposition import PCA


# args = ConfigNamepace.from_yaml_path('ddpm/dynamical/args_scheduled_linear.yaml')
args = ConfigNamepace.from_yaml_path(sys.argv[1])



device = 'cuda'

num_neurons = args.neurons

W_target = torch.randn(num_neurons, num_neurons, requires_grad=True, device=device)

# Make initial weights stable
with torch.no_grad():
    eigvals = torch.linalg.eigvals(W_target)
    spectral_radius = eigvals.abs().max()
    scale = 1.15
    W_target.data = W_target.data / (scale * spectral_radius)




# Create save directory - use same directory as load_path
save_base = f'ddpm/dynamical/logs/recovery_linear/run_{args.model_type}'
_, save_dir, _ = configure_logging_paths(save_base, [])

args.write_to_yaml(os.path.join(save_dir, 'args.yaml'))



if args.model_type == 'moment_matching':

    # Initialize single learnable weight matrix
    print("Initializing learnable weights...")
    W_learnable = torch.randn(num_neurons, num_neurons, requires_grad=True, device=device)

    # Make initial weights stable
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(W_learnable)
        spectral_radius = eigvals.abs().max()
        scale = 1.5
        W_learnable.data = W_learnable.data / (scale * spectral_radius)

    # Setup optimizer
    optimizer = optim.Adam([W_learnable], lr=1e-3)


elif args.model_type == 'ddpm':

    print("Initializing DDPM...")

    residual_model = DynamicalSystemTransitionModel(
        state_space_size = num_neurons,
        input_size = num_neurons,
        time_embedding_size = None,
        k = args.k,
        n = args.n,
        tau = args.tau,
        init_spectral_radius = 0.6,
    )

    input_model = InputModelBlock(
        sensory_shape=[num_neurons], network_input_size=num_neurons, device = device
    )

    assert args.duration_no_stim == 0.0, "Not ready for 2 epoch recovery with DDPM"
    sigma2x_schedule = torch.linspace(0.05, 0.05, int(args.duration / args.dt))

    ddpm_model = LinearSubspaceTeacherForcedHVAEReverseProcess(
        seperate_output_neurons = False,
        stabilise_nullspace = True,
        sample_ambient_dim = num_neurons,
        sample_shape = [num_neurons],
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = 2,
        device = device,
        euler_alpha = args.dt/args.tau,
        noise_scaler=args.noise_scaler,
        train_as_rnn=False,
    ).to(device)

    optimizer = optim.Adam(ddpm_model.parameters(), lr=1e-3)

    mse_key = "epsilon_hat"
    mse_key_target = 'kernel_target'


    # For transparency
    sch_fig, sch_axes = plt.subplots(2, 1, figsize=(5, 8))
    sch_axes[0].plot(
        ddpm_model.sigma2xt_schedule.cpu().numpy(), label="sigma2xt_schedule", alpha=0.4
    )
    sch_axes[0].plot(ddpm_model.a_t_schedule.cpu().numpy(), label="a_t_schedule", alpha=0.4)
    sch_axes[0].plot(
        ddpm_model.root_b_t_schedule.cpu().numpy(), label="root_b_t_schedule", alpha=0.4
    )
    sch_axes[0].plot(
        ddpm_model.noise_scaler_schedule.cpu().numpy(),
        label="noise_scaler_schedule",
        alpha=0.4,
    )
    sch_axes[0].plot(
        ddpm_model.base_samples_scaler_schedule.cpu().numpy(),
        label="base_samples_scaler_schedule",
        alpha=0.4,
    )
    sch_axes[0].plot(
        ddpm_model.residual_scaler_schedule.cpu().numpy(),
        label="residual_scaler_schedule",
        alpha=0.4,
    )
    sch_axes[0].legend()

    sch_axes[1].set_title("Time embeddings")
    sch_axes[1].imshow(ddpm_model.time_embeddings.time_embs.detach().cpu().numpy().T)

    plt.savefig(os.path.join(save_dir, "sigma_schedule_unrolling.png"))




# Pre-compute constants outside training loop
timesteps_stim = int(args.duration / args.dt)
timesteps_no_stim = int(args.duration_no_stim / args.dt)
timesteps = timesteps_stim + timesteps_no_stim

# Pre-compute noise schedule
initial_sigma2_xi = args.sigma2_xi_preschedule
initial_Sigma_xi = torch.eye(num_neurons, device=device) * initial_sigma2_xi
initial_Sigma_xi_chol = initial_Sigma_xi.sqrt()

if timesteps_no_stim > 0:
    sigma2_xis = torch.linspace(args.sigma2_xi_start, args.sigma2_xi_end, timesteps_no_stim)
    Sigma_xis = [torch.eye(num_neurons, device=device) * s for s in sigma2_xis]
    Sigma_xi_chols = [S.sqrt() for S in Sigma_xis]
else:
    Sigma_xis = []
    Sigma_xi_chols = []

# Training loop
print("Starting optimization...")
losses = []
mean_losses = []
cov_losses = []
time_losses = []  # For DDPM: store loss per time step

rel_error_means = []
rel_error_stds = []
rel_error_epochs = []

batch_size = args.batch_size

for epoch in tqdm(range(args.num_epochs)):
    optimizer.zero_grad()
    
    # Procedurally generate new targets for this epoch - all batch items at once
    h_batch = torch.zeros(batch_size, num_neurons, device=device)
    for b in range(batch_size):
        # Generate different input for each batch item
        if args.h_scale > 0.0:
            h_scales = torch.ones(num_neurons, device=device)
            h_scales.exponential_(lambd=1 / float(args.h_scale))
            h_batch[b] = torch.randn_like(h_scales) * h_scales
        else:
            h_batch[b] = torch.zeros(num_neurons, device=device).float()
    
    # Generate empirical moments for all batch items at once
    empirical_mean, empirical_cov, u_target_samples = generate_empirical_moments_batch(
        W_target, h_batch, args, timesteps, timesteps_stim, timesteps_no_stim, 
        initial_Sigma_xi_chol, Sigma_xi_chols)
    
    if args.model_type == 'moment_matching':
        # Generate moment predictions with learnable weights
        pred_mean, pred_cov = predict_moments(
            W_learnable, h_batch, args, timesteps, timesteps_stim, timesteps_no_stim, 
            initial_Sigma_xi, Sigma_xis)
        
        # Compute loss
        mean_loss = torch.mean((pred_mean - empirical_mean) ** 2)
        cov_loss = torch.mean((pred_cov - empirical_cov) ** 2)
        total_loss = mean_loss + cov_loss

    elif args.model_type == 'ddpm':

        forward_process = ddpm_model.noise(x_0=u_target_samples)

        reshaped_h_batch = h_batch.unsqueeze(1).repeat_interleave(args.K, 1)

        epsilon_hat_dict = ddpm_model.residual(
            x_samples=forward_process["x_t"].to(device),
            network_input=reshaped_h_batch.to(device),
        )
        residual_mse = torch.square(epsilon_hat_dict[mse_key] - forward_process[mse_key_target]).sum(-1)  # [batch, samples, time]

        total_loss_per_time = residual_mse.mean(0).mean(0)  # [time]
        total_loss = total_loss_per_time.mean()

        if total_loss.isnan() or total_loss.isinf():
            raise Exception
        
        # Store time-resolved losses for DDPM
        time_losses.append(total_loss_per_time.detach().cpu().numpy())
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    losses.append(total_loss.item())
    if args.model_type == 'moment_matching':
        mean_losses.append(mean_loss.item())
        cov_losses.append(cov_loss.item())
    else:  # ddpm
        mean_losses.append(0.0)  # Not used for DDPM
        cov_losses.append(0.0)   # Not used for DDPM
    
    if epoch % 1000 == 0:
        if args.model_type == 'moment_matching':
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, Mean Loss = {mean_loss.item():.6f}, Cov Loss = {cov_loss.item():.6f}")
        else:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}")
    
    # Save intermediate results every 10 epochs
    if epoch % 50 == 0:

        print(f"Saving intermediate results at epoch {epoch}...")
        
        # 1. Loss graphs
        if args.model_type == 'moment_matching':
            # Original moment matching plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Total loss - linear and log scale
            axes[0, 0].plot(losses[:epoch+1])
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_title('Total Loss (Linear Scale)')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(losses[:epoch+1])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Total Loss')
            axes[0, 1].set_title('Total Loss (Log Scale)')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
            
            # Mean and covariance losses
            axes[1, 0].plot(mean_losses[:epoch+1], label='Mean Loss')
            axes[1, 0].plot(cov_losses[:epoch+1], label='Covariance Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Component Losses (Linear Scale)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(mean_losses[:epoch+1], label='Mean Loss')
            axes[1, 1].plot(cov_losses[:epoch+1], label='Covariance Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Component Losses (Log Scale)')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
        elif args.model_type == 'ddpm':
            # DDPM plots with time-resolved losses
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Total loss over epochs
            axes[0, 0].plot(losses[:epoch+1])
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_title('Total Loss (Linear Scale)')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(losses[:epoch+1])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Total Loss')
            axes[0, 1].set_title('Total Loss (Log Scale)')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
            
            # Time-resolved losses: each timestep over epochs
            if len(time_losses) > 0:
                time_losses_array = np.array(time_losses)  # [epochs, timesteps]
                num_timesteps = time_losses_array.shape[1]
                epoch_array = np.arange(len(time_losses)) * 10  # Multiply by 10 since we save every 10 epochs
                
                # Plot each timestep as a separate line, colored by time
                cmap = plt.cm.viridis
                colors = [cmap(t / max(1, num_timesteps-1)) for t in range(num_timesteps)]
                
                for t in range(0, num_timesteps, max(1, num_timesteps//20)):  # Show max 20 lines
                    time_val = t * args.dt
                    axes[1, 0].plot(epoch_array, time_losses_array[:, t], 
                                   color=colors[t], alpha=0.7, 
                                   label=f't={time_val:.3f}s' if t % (num_timesteps//5) == 0 else '')
                
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Loss per Timestep Over Training')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                # Same plot as left but with log scale
                for t in range(0, num_timesteps, max(1, num_timesteps//20)):  # Show max 20 lines
                    time_val = t * args.dt
                    axes[1, 1].plot(epoch_array, time_losses_array[:, t], 
                                   color=colors[t], alpha=0.7, 
                                   label=f't={time_val:.3f}s' if t % (num_timesteps//5) == 0 else '')
                
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss (log scale)')
                axes[1, 1].set_title('Loss per Timestep Over Training (Log Scale)')
                axes[1, 1].set_yscale('log')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/losses_{args.model_type}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Output scatter comparison for first batch item
        with torch.no_grad():
            # Use the input from first batch item for consistent comparison
            h_comparison = h_batch[0]
            
            # Generate target outputs 
            target_final_states = u_target_samples[0]  # [trials, state_dim]
            
            # Generate learned outputs based on model type  [trials, state_dim]
            if args.model_type == 'moment_matching':
                learned_final_states = generate_learned_traj(W_learnable, h_comparison, args, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi_chol, Sigma_xi_chols)
            elif args.model_type == 'ddpm':
                # Use the learned W matrix from the residual model
                W_learned_ddpm = ddpm_model.residual_model.get_W_matrix()
                learned_final_states = generate_learned_traj(W_learned_ddpm, h_comparison, args, timesteps, timesteps_stim, timesteps_no_stim, initial_Sigma_xi_chol, Sigma_xi_chols)

            # Do PCA on target_final_states using sklearn

            # Move to cpu and numpy for sklearn
            target_np = target_final_states.detach().cpu().numpy()
            learned_np = learned_final_states.detach().cpu().numpy()

            pca = PCA(n_components=2, whiten=True)
            target_outputs_np = pca.fit_transform(target_np)
            learned_outputs_np = pca.transform(learned_np)

            # Convert back to torch tensors on original device
            target_outputs = torch.tensor(target_outputs_np, device=target_final_states.device, dtype=target_final_states.dtype)
            learned_outputs = torch.tensor(learned_outputs_np, device=learned_final_states.device, dtype=learned_final_states.dtype)

            
            # Plot scatter comparison
            output_dim = target_outputs.shape[-1]
            if output_dim >= 2:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.scatter(target_outputs[:, 0].cpu(), target_outputs[:, 1].cpu(), 
                            alpha=0.6, s=20, label='Target Network', color='red')
                ax.scatter(learned_outputs[:, 0].cpu(), learned_outputs[:, 1].cpu(), 
                            alpha=0.6, s=20, label=f'Learned Network ({args.model_type})', color='blue')
                ax.set_xlabel('Output Dimension 0')
                ax.set_ylabel('Output Dimension 1')
                if output_dim == 2:
                    ax.set_title(f'Network Outputs Comparison (Epoch {epoch})')
                else:
                    ax.set_title(f'Network Outputs Comparison (Epoch {epoch}) - First 2 Dims')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
            else:
                # For 1D output, make histogram comparison
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.hist(target_outputs[:, 0].cpu().numpy(), alpha=0.7, bins=30, label='Target Network', color='red')
                ax.hist(learned_outputs[:, 0].cpu().numpy(), alpha=0.7, bins=30, label=f'Learned Network ({args.model_type})', color='blue')
                ax.set_xlabel('Output Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Network Outputs Comparison (Epoch {epoch})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.savefig(f'{save_dir}/outputs_{args.model_type}.png', dpi=150, bbox_inches='tight')
            plt.close()

        
        # Plot W_learned and W_target matrices and their relative error
        if args.model_type == 'moment_matching':
            W_learned_plot = W_learnable.detach().cpu().numpy()
        elif args.model_type == 'ddpm':
            W_learned_plot = ddpm_model.residual_model.get_W_matrix().detach().cpu().numpy()
        W_target_plot = W_target.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axes[0].imshow(W_target_plot, aspect='auto', cmap='viridis')
        axes[0].set_title('W_target')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(W_learned_plot, aspect='auto', cmap='viridis')
        axes[1].set_title('W_learned')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Avoid division by zero in relative error
        denom = np.where(np.abs(W_target_plot) > 1e-8, np.abs(W_target_plot), 1e-8)
        rel_error = np.abs(W_learned_plot - W_target_plot) / denom
        im2 = axes[2].imshow(rel_error, aspect='auto', cmap='magma')
        axes[2].set_title('Relative Error |W_learned - W_target| / |W_target|')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/W_comparison_{args.model_type}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # --- Track and plot relative error over time ---
        # Compute and store current mean and std of relative error
        rel_error_mean = rel_error.mean()
        rel_error_std = rel_error.std()
        rel_error_means.append(rel_error_mean)
        rel_error_stds.append(rel_error_std)
        rel_error_epochs.append(epoch)

        # Plot the mean and 1 std band of relative error over epochs
        fig, ax = plt.subplots(figsize=(8, 6))
        rel_error_means_np = np.array(rel_error_means)
        rel_error_stds_np = np.array(rel_error_stds)
        rel_error_epochs_np = np.array(rel_error_epochs)
        ax.plot(rel_error_epochs_np, rel_error_means_np, label='Mean Relative Error', color='blue')
        # ax.fill_between(
        #     rel_error_epochs_np,
        #     rel_error_means_np - rel_error_stds_np,
        #     rel_error_means_np + rel_error_stds_np,
        #     color='blue', alpha=0.2, label='±1 Std'
        # )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative Error')
        ax.set_title('Relative Error of W_learned vs W_target Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/W_rel_error_over_time_{args.model_type}.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Save results based on model type
        if args.model_type == 'moment_matching':
            results = {
                'W_learned': W_learnable.detach().cpu(),
                'W_target': W_target.cpu(),
                'losses': losses,
                'mean_losses': mean_losses,
                'cov_losses': cov_losses,
                'args': args
            }
        elif args.model_type == 'ddpm':
            results = {
                'ddpm_model_state': ddpm_model.state_dict(),
                'W_learned': ddpm_model.residual_model.get_W_matrix().detach().cpu(),
                'W_target': W_target.cpu(),
                'losses': losses,
                'time_losses': time_losses,
                'args': args
            }
        
        weights_save_path = f'{save_dir}/state_{args.model_type}.mdl'
        torch.save(results, weights_save_path)
        print(f"Saved results to: {weights_save_path}")
