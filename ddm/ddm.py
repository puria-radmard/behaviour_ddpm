import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def generate_correct_answers(batch_size, p_upper=0.5):
    """
    Generate correct answers for each batch item.
    
    Args:
        batch_size: Number of batch items
        p_upper: Probability of correct answer being +1 (upper boundary)
    
    Returns:
        correct_answers: Tensor of shape [batch] with values in {+1, -1}
    """
    # Generate binary choices (1 or 0)
    binary_choices = torch.bernoulli(torch.full((batch_size,), p_upper))
    
    # Convert to +1/-1
    correct_answers = 2 * binary_choices - 1
    
    return correct_answers


def generate_sensory_batch(correct_answers, mu):
    """
    Generate a batch of sensory inputs based on correct answers.
    
    Args:
        correct_answers: Tensor of shape [batch] with values in {+1, -1}
        mu: Drift rate magnitude (scalar or tensor of shape [batch, trials])
        n_trials: Number of trials per batch item
    
    Returns:
        sensory_inputs: Tensor of shape [batch, trials]
    """
    batch_size, n_trials = mu.shape
    
    # Simple rule for now: sensory_input = correct_answer * mu
    # Expand to [batch, trials]
    sensory_inputs = correct_answers.unsqueeze(1) * mu
    sensory_inputs = sensory_inputs.expand(batch_size, n_trials)

    return sensory_inputs


def simulate_ddm(sensory_inputs, max_steps, dt=0.001, 
                 sigma=1.0, boundary=1.0, starting_point=0.0, *_, use_cum_sum: bool = False):
    """
    Simulate drift diffusion model trajectories.
    
    Args:
        sensory_inputs: Tensor of shape [batch, trials] containing drift rates
        max_steps: Maximum number of time steps
        dt: Discretization time step
        sigma: Diffusion coefficient (noise strength)
        boundary: Decision boundary (symmetric at ±boundary)
        starting_point: Initial evidence value
    
    Returns:
        trajectories: Tensor of shape [batch, trials, time_steps]
        choices: Tensor of shape [batch, trials] with 1 for upper, -1 for lower boundary
        reaction_times: Tensor of shape [batch, trials] with RT in time steps
        completed: Boolean tensor of shape [batch, trials] indicating completed trials
    """
    batch_size, n_trials = sensory_inputs.shape
    device = sensory_inputs.device


    
    # Initialize trajectories
    trajectories = torch.zeros(batch_size, n_trials, max_steps, device=device)
    trajectories[:, :, 0] = starting_point
    
    # Initialize tracking variables
    completed = torch.zeros(batch_size, n_trials, dtype=torch.bool, device=device)
    choices = torch.zeros(batch_size, n_trials, dtype=torch.long, device=device)
    reaction_times = torch.full((batch_size, n_trials), max_steps * dt, dtype=torch.float, device=device)
    
    # Scaling factors for discrete-time simulation
    drift_scaled = sensory_inputs * dt
    noise_scaled = sigma * torch.sqrt(torch.tensor(dt, device=device))


    if use_cum_sum:
        # Vectorized simulation using cumulative sum
        # Generate all noise at once
        all_noise = torch.randn(batch_size, n_trials, max_steps-1, device=device) * noise_scaled
        
        # Create time indices for drift contribution
        time_indices = torch.arange(1, max_steps, device=device).float() * dt
        drift_contribution = sensory_inputs.unsqueeze(-1) * time_indices
        
        # Compute full trajectories using cumsum
        noise_contribution = torch.cumsum(all_noise, dim=2)
        trajectories[:, :, 1:] = starting_point + drift_contribution + noise_contribution
        
        # Find boundary crossings
        upper_crossings = trajectories >= boundary
        lower_crossings = trajectories <= -boundary
        any_crossing = upper_crossings | lower_crossings
        
        # Find first crossing time for each trial (argmax returns 0 if no True values)
        first_crossing_times = torch.argmax(any_crossing.float(), dim=2)
        
        # Check if any crossing actually occurred
        has_crossing = any_crossing.any(dim=2)
        
        # For trials with no crossing, set to max_steps-1
        first_crossing_times[~has_crossing] = max_steps - 1
        
        # Get choices based on the value at first crossing time
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        trial_idx = torch.arange(n_trials, device=device).unsqueeze(0)
        crossing_values = trajectories[batch_idx, trial_idx, first_crossing_times]
        
        choices = torch.where(crossing_values >= boundary, 1, -1)
        choices[~has_crossing] = 0  # No decision for uncompleted trials
        
        # Set reaction times
        reaction_times = first_crossing_times.float() * dt
        reaction_times[~has_crossing] = max_steps * dt
        
        # Clamp trajectories after boundary crossing
        time_grid = torch.arange(max_steps, device=device).view(1, 1, -1)
        after_crossing = time_grid > first_crossing_times.unsqueeze(-1)
        
        # Set post-crossing values to the boundary that was hit
        hit_upper_boundary = (crossing_values >= boundary).unsqueeze(-1) & after_crossing
        hit_lower_boundary = (crossing_values <= -boundary).unsqueeze(-1) & after_crossing
        
        trajectories[hit_upper_boundary] = boundary
        trajectories[hit_lower_boundary] = -boundary
        
        completed = has_crossing
        
    else:
        # Simulate each time step
        for t in range(1, max_steps):
            # Only update trials that haven't completed
            active_mask = ~completed
            
            if not active_mask.any():
                break
                
            # Generate noise
            noise = torch.randn(batch_size, n_trials, device=device) * noise_scaled
            
            # Update evidence for active trials
            trajectories[:, :, t] = trajectories[:, :, t-1].clone()
            trajectories[:, :, t][active_mask] += drift_scaled[active_mask] + noise[active_mask]
            
            # Check boundary crossings
            hit_upper = (trajectories[:, :, t] >= boundary) & active_mask
            hit_lower = (trajectories[:, :, t] <= -boundary) & active_mask
            
            # Record choices and RTs for newly completed trials
            newly_completed = hit_upper | hit_lower
            choices[hit_upper] = 1
            choices[hit_lower] = -1
            reaction_times[newly_completed] = t * dt
            
            # Update completed mask
            completed |= newly_completed
            
            # Clamp trajectories at boundaries for completed trials
            trajectories[:, :, t] = torch.clamp(trajectories[:, :, t], -boundary, boundary)
        
    return trajectories, choices, reaction_times, completed


def analyze_ddm_results(correct_answers, choices, reaction_times, completed):
    """
    Analyze DDM simulation results.
    
    Args:
        correct_answers: Tensor of shape [batch] with values in {+1, -1}
        choices: Tensor of shape [batch, trials] with values in {1, -1}
        reaction_times: Tensor of shape [batch, trials]
        completed: Boolean tensor of shape [batch, trials]
    
    Returns dictionary with:
        - accuracy: Proportion correct for each batch item
        - rt_correct: List of RT arrays for correct responses
        - rt_incorrect: List of RT arrays for incorrect responses
    """
    batch_size = choices.shape[0]
    n_trials = choices.shape[1]
    
    # Expand correct answers for comparison
    correct_answers_expanded = correct_answers.unsqueeze(1).expand(batch_size, n_trials)
    
    # Identify correct trials. Incomplete trials are zero anyway
    correct = (choices == correct_answers_expanded)
    incorrect = (choices == -correct_answers_expanded)
    
    # Calculate accuracy
    accuracy = correct.sum(dim=1).float() / completed.sum(dim=1).float()
    
    # Collect RTs for each batch item
    rt_correct = []
    rt_incorrect = []

    for i in range(batch_size):
        rt_correct.append(reaction_times[i][correct[i]].cpu().numpy())
        rt_incorrect.append(reaction_times[i][incorrect[i]].cpu().numpy())
    
    return {
        'accuracy': accuracy,
        'rt_correct': rt_correct,
        'rt_incorrect': rt_incorrect,
        'completion': completed.float().mean(dim = 1).float()
    }


def plot_ddm_results(mu_values, trajectories, results, dt, boundary, save_path='ddm'):
    """
    Create figure with trajectories and RT histograms for each mu value.
    """
    batch_size = len(mu_values)
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 3*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    time_axis = torch.arange(trajectories.shape[2]) * dt
    
    for i, mu in enumerate(mu_values):
        # Left column: trajectory plots
        ax_traj = axes[i, 0]
        
        # Plot first 10 trajectories
        traj_subset = trajectories[i, :10, :].T.cpu().numpy()
        ax_traj.plot(time_axis.cpu().numpy(), traj_subset, alpha=0.5, linewidth=0.8)
        ax_traj.axhline(y=boundary, color='r', linestyle='--', label='Upper boundary')
        ax_traj.axhline(y=-boundary, color='b', linestyle='--', label='Lower boundary')
        ax_traj.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax_traj.set_xlim(0, time_axis[-1].item())
        ax_traj.set_ylim(-1.0, 1.0)
        ax_traj.set_xlabel('Time (s)')
        ax_traj.set_ylabel('Evidence')
        ax_traj.set_title(f'μ = {mu:.1f} - Example Trajectories')
        ax_traj.grid(True, alpha=0.3)
        
        if i == 0:
            ax_traj.legend()
        
        # Right column: RT histograms
        ax_hist = axes[i, 1]
        
        rt_correct = results['rt_correct'][i]
        rt_incorrect = results['rt_incorrect'][i]
        
        # Create histograms
        if len(rt_correct) > 0:
            ax_hist.hist(rt_correct, bins=50, alpha=0.6, color='green', 
                        label=f'Correct (n={len(rt_correct)})', density=True)
        
        if len(rt_incorrect) > 0:
            ax_hist.hist(rt_incorrect, bins=50, alpha=0.6, color='red', 
                        label=f'Incorrect (n={len(rt_incorrect)})', density=True)
        
        # Plot vertical lines for mean RTs
        if len(rt_correct) > 0:
            mean_rt_correct = rt_correct.mean()
            ax_hist.axvline(mean_rt_correct, color='green', linestyle='--', linewidth=2,
                            label=f'Mean Correct RT: {mean_rt_correct:.2f}s')
            ax_hist.text(mean_rt_correct, ax_hist.get_ylim()[1]*0.9, f'{mean_rt_correct:.2f}s',
                         color='green', rotation=90, va='top', ha='right', fontsize=9, backgroundcolor='white')

        if len(rt_incorrect) > 0:
            mean_rt_incorrect = rt_incorrect.mean()
            ax_hist.axvline(mean_rt_incorrect, color='red', linestyle='--', linewidth=2,
                            label=f'Mean Incorrect RT: {mean_rt_incorrect:.2f}s')
            ax_hist.text(mean_rt_incorrect, ax_hist.get_ylim()[1]*0.7, f'{mean_rt_incorrect:.2f}s',
                         color='red', rotation=90, va='top', ha='right', fontsize=9, backgroundcolor='white')
        
        ax_hist.set_xlabel('Reaction Time (s)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'μ = {mu:.1f} - RT Distributions (Acc: {results["accuracy"][i]:.3f}, Comp: {results["completion"][i]:.3f})')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_xlim(0, time_axis[-1].item())
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(save_path, 'ddm_results.png'), dpi=150, bbox_inches='tight')


# Example usage
if __name__ == "__main__":
    
    # Parameters
    n_trials = 2048
    max_dur = 5.0
    dt = 0.001
    
    max_steps = int(max_dur / dt)
    
    # DDM parameters
    mu_values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]).unsqueeze(1).repeat(1, n_trials)  # Different drift rates
    sigma = 1.0
    boundary = 1.0

    batch_size = len(mu_values)
    
    # Generate correct answers (all upper boundary for this example)
    correct_answers = generate_correct_answers(batch_size)  # All +1 (upper boundary)
    
    # Generate sensory inputs
    sensory_inputs = generate_sensory_batch(correct_answers, mu_values)
    
    # Run simulation
    trajectories, choices, reaction_times, completed = simulate_ddm(
        sensory_inputs, max_steps, dt, sigma, boundary
    )

    # Analyze results
    results = analyze_ddm_results(correct_answers, choices, reaction_times, completed)
    
    # Print results
    print("DDM Simulation Results:")
    print("-" * 50)
    for i, mu in enumerate(mu_values):
        print(f"\nDrift rate μ = {mu}:")
        print(f"  Accuracy: {results['accuracy'][i]:.3f}")
        print(f"  Correct RTs: n={len(results['rt_correct'][i])}, "
              f"mean={results['rt_correct'][i].mean():.1f} steps" if len(results['rt_correct'][i]) > 0 else "no correct trials")
        print(f"  Incorrect RTs: n={len(results['rt_incorrect'][i])}, "
              f"mean={results['rt_incorrect'][i].mean():.1f} steps" if len(results['rt_incorrect'][i]) > 0 else "no incorrect trials")
        print(f"  Completed trials: {completed[i].sum().item()}/{n_trials}")
    
    # Create visualization
    plot_ddm_results(mu_values, trajectories, results, dt, boundary, save_path='ddm')

    