import torch
import math
from ddpm.dynamical.training import DynamicalNetwork, working_memory_task
import matplotlib.pyplot as plt


if __name__ == '__main__':

    from purias_utils.util.arguments_yaml import ConfigNamepace
    args = ConfigNamepace.from_yaml_path('ddpm/dynamical/args_trained.yaml')

    # Parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Task parameters
    num_mems=args.num_mems
    
    # Network parameters
    neurons = args.neurons
    num_in_dims = num_mems + 1
    num_out_dims = 2
    batch_size = args.batch_size
    trials = 1
    
    # Dynamics parameters
    dt = args.dt
    tau = args.tau
    k = args.k
    n = args.n
    noise_scale = args.sigma2_xi

    with torch.no_grad():

        # Initialize network and optimizer ONCE
        net = DynamicalNetwork(neurons, num_in_dims, num_out_dims).to(device)
        net.load_state_dict(torch.load('ddpm/dynamical/logs/training/net.mdl'))
        Sigma_xi = torch.eye(neurons, device=device) * noise_scale

        # Generate fresh batch of task data
        h_sequence, r_target = working_memory_task(
            num_mems=num_mems,
            device=device
        )
        
        # Forward pass
        r_pred = net(h_sequence, trials, dt, tau, k, n, Sigma_xi)
        
        # Plot results for first batch item of current training data
        fig, axes = plt.subplots(3, 3, figsize=(25, 10))
        axes = axes.T

        for b in range(3):
        
            # Plot inputs (flip commands)
            for i in range(num_in_dims):
                axes[b, 0].plot(h_sequence[b, :, i].cpu(), label=f'Flip_{i}', alpha=0.7)
            axes[b, 0].set_ylabel('Flip Signal')
            axes[b, 0].legend()
            axes[b, 0].grid(True, alpha=0.3)
            
            # Plot targets vs predictions
            for i in range(num_out_dims):
                tar_line_col = axes[b, 1].plot(r_target[b, :, i].cpu(), '--', label=f'Target_{i}', alpha=0.8, linewidth=2)[0].get_color()
                axes[b, 1].plot(r_pred[b, 0, :, i].detach().cpu(), '-', label=f'Pred_{i}', alpha=0.8, color = tar_line_col)
            axes[b, 1].set_title('Flip-Flop Memory: Target vs Prediction')
            axes[b, 1].set_ylabel('Memory State')
            axes[b, 1].legend()
            axes[b, 1].grid(True, alpha=0.3)
            
            # Plot error
            error = (r_pred[b,0] - r_target[b]).abs()
            for i in range(num_out_dims):
                axes[b, 2].plot(error[:, i].detach().cpu(), label=f'|Error|_{i}', alpha=0.7)
            axes[b, 2].set_title('Absolute Error')
            axes[b, 2].set_xlabel('Time Step')
            axes[b, 2].set_ylabel('|Prediction - Target|')
            axes[b, 2].legend()
            axes[b, 2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('ddpm/dynamical/logs/prediction/flip_flop_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
