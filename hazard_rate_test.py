import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

# Import the utilities from previous artifact
import math

from ddpm.utils.hazard_rate import (
    log_normal_to_discrete, 
    response_time_loss,
    sample_response_times_batch,
    fit_lognormal_parameters
)


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class LeakyRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dt: float, tau: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.alpha = dt / tau  # Integration constant
        
        # Input and recurrent weights
        self.input_proj = nn.Linear(input_size, hidden_size, bias=True)
        self.recurrent_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, inputs: torch.Tensor, warmup_steps: int, response_steps: int, num_trials: int = 1):
        """
        Args:
            inputs: [batch_size, input_size] - constant inputs
            warmup_steps: number of warmup timesteps
            response_steps: number of response timesteps
            num_trials: number of independent trials to simulate
        
        Returns:
            hazard_rates: [batch_size, num_trials, response_steps] - hazard rates during response period
        """
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Expand inputs for multiple trials: [batch_size, input_size] -> [batch_size * num_trials, input_size]
        inputs_expanded = inputs.unsqueeze(1).expand(-1, num_trials, -1).reshape(-1, inputs.size(1))
        expanded_batch_size = batch_size * num_trials
        
        # Initialize hidden state for all trials
        h = torch.zeros(expanded_batch_size, self.hidden_size, device=device)
        
        # Project constant inputs once
        input_contribution = self.input_proj(inputs_expanded)  # [batch_size * num_trials, hidden_size]
        
        # Warmup period - no output collection, avoid in-place operations
        for _ in range(warmup_steps):
            recurrent_contribution = self.recurrent_proj(F.softplus(h))
            h = h * (1 - self.alpha) + self.alpha * (recurrent_contribution + input_contribution)
        
        # Response period - collect hidden states for batch processing
        hidden_states = []
        for _ in range(response_steps):
            recurrent_contribution = self.recurrent_proj(F.softplus(h))
            h = h * (1 - self.alpha) + self.alpha * (recurrent_contribution + input_contribution)
            hidden_states.append(h)
        
        # Stack all hidden states and process through output layer at once
        # [response_steps, batch_size * num_trials, hidden_size] -> [batch_size * num_trials * response_steps, hidden_size]
        stacked_hidden = torch.stack(hidden_states, dim=0)  # [response_steps, batch_size * num_trials, hidden_size]
        stacked_hidden = stacked_hidden.transpose(0, 1).reshape(-1, self.hidden_size)  # [batch_size * num_trials * response_steps, hidden_size]
        
        # Process all outputs at once
        outputs = self.output_proj(stacked_hidden)  # [batch_size * num_trials * response_steps, 1]
        hazard_rates = torch.sigmoid(outputs)  # [batch_size * num_trials * response_steps, 1]
        
        # Reshape to final form: [batch_size, num_trials, response_steps]
        hazard_rates = hazard_rates.reshape(batch_size, num_trials, response_steps)
        
        return hazard_rates


def generate_batch(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a new batch of log-normal parameters."""
    # Sample log-normal parameters
    log_means = torch.rand(batch_size, device=device) * 1.3 - 0.5  # Uniform(-0.5, 0.8)
    log_vars = torch.rand(batch_size, device=device) * 0.5 + 0.3   # Uniform(0.3, 0.8)
    
    # Create inputs (the parameters themselves)
    inputs = torch.stack([log_means, log_vars], dim=1)  # [batch_size, 2]
    
    return inputs, log_means, log_vars



def main():
    # Hyperparameters
    dt = 0.05
    tau = 10 * dt  # 0.5
    warmup_steps = 5
    response_steps = 64
    hidden_size = 64
    batch_size = 64
    num_trials = 512
    total_iterations = 1000000
    plot_every = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = LeakyRNN(input_size=2, hidden_size=hidden_size, dt=dt, tau=tau).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training history
    loss_history = []
    
    print("Starting training...")
    
    for iteration in range(total_iterations):
        # Generate new batch for each iteration!
        inputs, log_means, log_vars = generate_batch(batch_size, device)
        
        # Forward pass through model for all trials
        all_hazard_rates = model(inputs, warmup_steps, response_steps, num_trials)  # [batch, num_trials, response_steps]
        
        # Compute loss
        losses = response_time_loss(all_hazard_rates, log_means, log_vars, dt)
        loss = losses.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Plotting
        if iteration % plot_every == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
            
            # Sample response times from current training batch for parameter fitting
            with torch.no_grad():
                train_response_times = sample_response_times_batch(all_hazard_rates, samples_per_batch=1000)
                fitted_log_means, fitted_log_vars = fit_lognormal_parameters(train_response_times, dt)
                fitted_log_means_filtered, fitted_log_vars_filtered = fit_lognormal_parameters(
                    train_response_times, dt, exclude_max=True, max_timesteps=response_steps
                )
            
            # Generate test data for response time histogram
            with torch.no_grad():
                test_inputs, test_log_means, test_log_vars = generate_batch(1, device)
                test_hazard_rates = model(test_inputs, warmup_steps, response_steps, 100)  # [1, 100, response_steps]
                
                # Sample response times for histogram
                test_response_times_batch = sample_response_times_batch(test_hazard_rates, samples_per_batch=1000)
                test_response_times = test_response_times_batch[0].cpu().numpy()  # Extract first (and only) batch element
                
                # Get target distribution
                target_dist = log_normal_to_discrete(
                    test_log_means, test_log_vars, dt, response_steps
                ).cpu().numpy()[0]
            
            # Create plots - now 6 subplots (2x3 grid)
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 8))
            
            # 1. Loss history with log scale
            ax1.plot(loss_history)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.set_yscale('log')
            ax1.grid(True)
            
            # 2. Response time histogram with target overlay
            time_bins = np.arange(response_steps + 1) - 0.5
            ax2.hist(test_response_times, bins=time_bins, density=True, alpha=0.7, 
                    label='Sampled RTs', color='skyblue')
            
            # Plot target distribution
            time_centers = np.arange(response_steps)
            ax2.plot(time_centers, target_dist, 'r-', linewidth=2, 
                    label='Target Distribution')
            
            ax2.set_xlabel('Response Time (timesteps)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title(f'Response Times (iter {iteration})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Scatter plot: fitted vs target log means (all samples)
            target_means_cpu = log_means.cpu().numpy()
            fitted_means_cpu = fitted_log_means.cpu().numpy()
            
            ax3.scatter(target_means_cpu, fitted_means_cpu, alpha=0.6, s=20)
            
            # Add diagonal line for perfect fit
            min_val = min(target_means_cpu.min(), fitted_means_cpu.min())
            max_val = max(target_means_cpu.max(), fitted_means_cpu.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
            
            ax3.set_xlabel('Target Log Means')
            ax3.set_ylabel('Fitted Log Means')
            ax3.set_title('Log Means: All Samples')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Scatter plot: fitted vs target log variances (all samples)
            target_vars_cpu = log_vars.cpu().numpy()
            fitted_vars_cpu = fitted_log_vars.cpu().numpy()
            
            ax4.scatter(target_vars_cpu, fitted_vars_cpu, alpha=0.6, s=20)
            
            # Add diagonal line for perfect fit
            min_val = min(target_vars_cpu.min(), fitted_vars_cpu.min())
            max_val = max(target_vars_cpu.max(), fitted_vars_cpu.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
            
            ax4.set_xlabel('Target Log Variances')
            ax4.set_ylabel('Fitted Log Variances')
            ax4.set_title('Log Variances: All Samples')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Scatter plot: fitted vs target log means (excluding max RT)
            fitted_means_filtered_cpu = fitted_log_means_filtered.cpu().numpy()
            
            ax5.scatter(target_means_cpu, fitted_means_filtered_cpu, alpha=0.6, s=20, color='orange')
            
            # Add diagonal line for perfect fit
            min_val = min(target_means_cpu.min(), fitted_means_filtered_cpu.min())
            max_val = max(target_means_cpu.max(), fitted_means_filtered_cpu.max())
            ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
            
            ax5.set_xlabel('Target Log Means')
            ax5.set_ylabel('Fitted Log Means')
            ax5.set_title('Log Means: Early Stops Only')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Scatter plot: fitted vs target log variances (excluding max RT)
            fitted_vars_filtered_cpu = fitted_log_vars_filtered.cpu().numpy()
            
            ax6.scatter(target_vars_cpu, fitted_vars_filtered_cpu, alpha=0.6, s=20, color='orange')
            
            # Add diagonal line for perfect fit
            min_val = min(target_vars_cpu.min(), fitted_vars_filtered_cpu.min())
            max_val = max(target_vars_cpu.max(), fitted_vars_filtered_cpu.max())
            ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
            
            ax6.set_xlabel('Target Log Variances')
            ax6.set_ylabel('Fitted Log Variances')
            ax6.set_title('Log Variances: Early Stops Only')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('hazard_rate_test.png', dpi=100)
            plt.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()