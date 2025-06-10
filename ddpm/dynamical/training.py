import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor as _T
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import functions from the dynamics module
from ddpm.dynamical.moment_pred import (
    f, du_dt
)

class DynamicalNetwork(nn.Module):
    def __init__(self, neurons: int, in_dim: int, out_dim: int):
        super().__init__()
        self.neurons = neurons
        
        # W is the trainable weight matrix
        self.W = nn.Parameter(torch.randn(neurons, neurons) * 0.1)
        self.in_weight = nn.Linear(in_dim, neurons)
        self.out_weight = nn.Linear(neurons, out_dim)
    
    def forward(self, h_sequence: _T, trials: int, dt: float, 
                tau: float, k: float, n: int, Sigma_xi: _T) -> _T:
        """
        Forward pass: simulate dynamics and compute output rates
        
        Args:
            h_sequence: [batch, time, in_dim] - input sequence
            
        Returns:
            output: [batch, trials, time, out_dim] - output rates
        """
        # Project input to neural space: [batch, time, neurons]
        h_neural = self.in_weight(h_sequence)
        
        batch, time_steps, neurons = h_neural.shape
        assert neurons == self.neurons
        
        # Noise setup
        Sigma_xi_chol = torch.cholesky(Sigma_xi)
        
        # Initial conditions: [batch, trials, neurons]
        u = torch.randn(batch, trials, neurons, device=self.W.device) * 0.1
        outputs = []
        
        for t in range(time_steps):
            h_t = h_neural[:, t, :]  # [batch, neurons]
            
            # Generate noise increment: [batch, trials, neurons]
            dxi = torch.randn(batch, trials, neurons, device=self.W.device) @ Sigma_xi_chol.T * math.sqrt(dt)
            
            # Update dynamics
            du = du_dt(u, h_t, tau, self.W[None,None], dt, dxi, k, n)
            u = u + du * dt
            
            # Compute firing rates and project to output space
            rates = f(u, k, n)  # [batch, trials, neurons]
            output = self.out_weight(rates)  # [batch, trials, out_dim]
            outputs.append(output)
        
        # Stack: [batch, trials, time, out_dim]
        return torch.stack(outputs, dim=2)

def flip_flop_task(batch_size: int = 4, time_steps: int = 100, num_flops: int = 3, 
                   grace_period: int = 10, minimum_flip_spacing: int = 3, 
                   flip_probability: float = 0.05, device: str = 'cpu'):
    """Generate flip-flop task data"""
    
    # Initialize tensors
    h_sequence = torch.zeros(batch_size, time_steps, num_flops, device=device)
    r_target = torch.full((batch_size, time_steps, num_flops), -1.0, device=device)
    
    for batch_idx in range(batch_size):
        # Track last flip time for each dimension to enforce minimum spacing
        last_flip_time = [-minimum_flip_spacing] * num_flops
        
        # Track current state for each flip-flop dimension
        current_state = [-1.0] * num_flops  # All start at -1
        
        for t in range(time_steps):
            # Set current state as target
            for flop_idx in range(num_flops):
                r_target[batch_idx, t, flop_idx] = current_state[flop_idx]
            
            # Skip grace period - no flips allowed
            if t < grace_period:
                continue
                
            # For each flip-flop dimension, decide whether to flip
            for flop_idx in range(num_flops):
                # Check if enough time has passed since last flip
                if t - last_flip_time[flop_idx] >= minimum_flip_spacing:
                    # Random chance to flip
                    if torch.rand(1).item() < flip_probability:
                        # Execute flip
                        h_sequence[batch_idx, t, flop_idx] = 1.0
                        current_state[flop_idx] *= -1.0  # Flip the state
                        last_flip_time[flop_idx] = t
    
    return h_sequence, r_target



def working_memory_task(batch_size: int = 4, num_mems: int = 8, 
                       prestim_steps: int = 10, stim_steps: int = 10,
                       delay_min: int = 10, delay_max: int = 50,
                       signal_steps: int = 3, min_response_steps: int = 20,
                       device: str = 'cpu'):
    """
    Working memory task with 5 phases:
    1. Prestim - no input
    2. Stim - one hot encoding on first num_mems elements, last element zero
    3. Delay - variable duration, no input  
    4. Signal - signal_steps duration, last element is 1
    5. Response - no input
    
    Args:
        batch_size: number of trials
        num_mems: number of memory items (network input size is num_mems + 1)
        prestim_steps: duration of prestimulus period
        stim_steps: duration of stimulus presentation
        delay_min/max: min/max delay duration (randomized per trial)
        signal_steps: duration of go signal (usually 3)
        response_steps: duration of response period
        
    Returns:
        h_sequence: [batch, time, num_mems + 1] - input sequence
        r_target: [batch, time, 2] - target output (2D unit circle)
    """
    
    # Calculate total time steps (delay varies per trial)
    max_time = prestim_steps + stim_steps + delay_max + signal_steps + min_response_steps
    
    # Initialize tensors
    h_sequence = torch.zeros(batch_size, max_time, num_mems + 1, device=device)
    r_target = torch.zeros(batch_size, max_time, 2, device=device)
    
    for batch_idx in range(batch_size):
        # Randomly select which memory item to present (1 to num_mems-1)
        stim_idx = torch.randint(1, num_mems, (1,)).item()
        
        # Randomly select delay duration
        delay_duration = torch.randint(delay_min, delay_max + 1, (1,)).item()
        
        # Calculate phase boundaries for this trial
        prestim_end = prestim_steps
        stim_end = prestim_end + stim_steps
        delay_end = stim_end + delay_duration
        signal_end = delay_end + signal_steps
        
        # Phase 1: Prestim (already zeros)
        
        # Phase 2: Stim - one hot encoding
        h_sequence[batch_idx, prestim_end:stim_end, stim_idx] = 1.0
        
        # Phase 3: Delay (already zeros)
        
        # Phase 4: Signal - last element is 1
        h_sequence[batch_idx, delay_end:signal_end, -1] = 1.0
        
        # Phase 5: Response - target output is point on unit circle
        angle = stim_idx / num_mems * 2 * math.pi
        target_x = math.cos(angle)
        target_y = math.sin(angle)
        
        r_target[batch_idx, signal_end:, 0] = target_x
        r_target[batch_idx, signal_end:, 1] = target_y
    
    return h_sequence, r_target




if __name__ == '__main__':
    # Set random seed for reproducibility
    smoothing_alpha = 0.4
    
    # Parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Task parameters
    num_mems=5
    
    # Network and training parameters
    neurons = 32
    num_in_dims = num_mems + 1
    num_out_dims = 2
    batch_size = 64
    time_steps = 1000
    trials = 1
    epochs = 10000000
    
    # Dynamics parameters
    dt = 0.001
    tau = 0.1
    k = 1.0
    n = 1
    noise_scale = 0.01
    lr = 0.005
    print(f"Training network: {neurons} neurons, {num_in_dims}→{num_out_dims} dims")
    print(f"Task: {batch_size} batches, {time_steps} steps, {trials} trials")
    
    # Initialize network and optimizer ONCE
    net = DynamicalNetwork(neurons, num_in_dims, num_out_dims).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    Sigma_xi = torch.eye(neurons, device=device) * noise_scale

    net.load_state_dict(torch.load('ddpm/dynamical/net.mdl'))
    optimizer.load_state_dict(torch.load('ddpm/dynamical/opt.mdl'))
    
    # Training loop
    losses = []
    smoothed_losses = [1.0]
    
    for epoch in tqdm(range(epochs)):
        # Generate fresh batch of task data
        h_sequence, r_target = working_memory_task(
            batch_size=batch_size,
            num_mems=num_mems,
            device=device
        )
        
        # Forward pass
        optimizer.zero_grad()
        r_pred = net(h_sequence, trials, dt, tau, k, n, Sigma_xi)
        
        # MSE loss, for each trial seperately
        loss = nn.functional.mse_loss(r_pred, r_target.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Log
        losses.append(loss.item())
        smoothed_losses.append(smoothing_alpha * smoothed_losses[-1] + (1.0 - smoothing_alpha) * losses[-1])
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")
            
            # Plot results for first batch item of current training data
            fig, axes = plt.subplots(3, 3, figsize=(25, 10))
            axes = axes.T

            for b in range(3):
            
                # Plot inputs (flip commands)
                for i in range(num_in_dims):
                    axes[b, 0].plot(h_sequence[b, :, i].cpu(), label=f'Flip_{i}', alpha=0.7)
                axes[b, 0].set_title(f'Flip Commands - Epoch {epoch}')
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
            plt.savefig('ddpm/dynamical/flip_flop_examples.png', dpi=150, bbox_inches='tight')
            plt.close()

            torch.save(net.state_dict(), 'ddpm/dynamical/net.mdl')
            torch.save(optimizer.state_dict(), 'ddpm/dynamical/opt.mdl')
    
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses, alpha = 0.3, color = 'blue')
        plt.plot(smoothed_losses[1:], color = 'blue')
        plt.title('Flip-Flop Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('ddpm/dynamical/flip_flop_training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    # Print learned weight matrix properties
    with torch.no_grad():
        W = net.W.cpu()
        eigvals = torch.linalg.eigvals(W)
        spectral_radius = eigvals.abs().max()
        print(f"Learned W spectral radius: {spectral_radius:.4f}")
        print(f"W matrix norm: {W.norm():.4f}")