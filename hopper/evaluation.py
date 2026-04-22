"""
Model evaluation and plotting utilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_rollout(model, X_test, Y_test, horizon=50, device='cpu'):
    """
    Evaluate model on multi-step rollout predictions.

    Args:
        model: Trained PyTorch model
        X_test: Test inputs [N, input_dim]
        Y_test: Test outputs (deltas) [N, output_dim]
        horizon: Number of steps to roll out
        device: PyTorch device

    Returns:
        rollout_results: Dict containing:
            - 'predictions': [horizon, output_dim] predicted trajectory
            - 'ground_truth': [horizon, output_dim] actual trajectory
            - 'errors': [horizon, output_dim] prediction errors
            - 'mse': Scalar mean squared error
    """
    model.eval()

    # Take first sample as initial condition
    x0 = X_test[0]
    state = x0[:6].copy()  # [pos(3), eul(3)]

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for t in range(min(horizon, len(X_test) - 1)):
            # Get current input
            x_t = X_test[t]

            # Predict delta
            x_tensor = torch.from_numpy(x_t).float().to(device).unsqueeze(0)
            delta_pred = model(x_tensor).cpu().numpy()[0]

            # Integrate to get next state
            state_pred = state + delta_pred

            # Ground truth next state
            state_true = state + Y_test[t]

            predictions.append(state_pred)
            ground_truth.append(state_true)

            # Update state for next iteration
            state = state_true  # Use ground truth for open-loop rollout

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    errors = predictions - ground_truth
    mse = np.mean(errors**2)

    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'errors': errors,
        'mse': mse
    }


def compute_metrics(model, dataloader, device='cpu'):
    """
    Compute evaluation metrics on a dataset.

    Args:
        model: Trained PyTorch model
        dataloader: PyTorch DataLoader
        device: PyTorch device

    Returns:
        metrics: Dict with 'loss', 'mae', 'mse'
    """
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward pass
            Y_pred = model(X_batch)

            # Compute losses
            mse_loss = torch.mean((Y_pred - Y_batch)**2)
            mae_loss = torch.mean(torch.abs(Y_pred - Y_batch))

            batch_size = X_batch.size(0)
            total_loss += mse_loss.item() * batch_size
            total_mae += mae_loss.item() * batch_size
            total_samples += batch_size

    metrics = {
        'mse': total_loss / total_samples,
        'mae': total_mae / total_samples,
        'loss': total_loss / total_samples,
    }

    return metrics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_history(train_losses, val_losses=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
    """
    plt.figure(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def plot_rollout_predictions(rollout_results, state_labels=None):
    """
    Plot rollout predictions vs. ground truth.

    Args:
        rollout_results: Dict from evaluate_rollout()
        state_labels: List of state dimension labels (optional)
    """
    predictions = rollout_results['predictions']
    ground_truth = rollout_results['ground_truth']
    horizon = predictions.shape[0]
    output_dim = predictions.shape[1]

    if state_labels is None:
        state_labels = [f'State {i}' for i in range(output_dim)]

    # Create subplots for each state dimension
    fig, axes = plt.subplots(output_dim, 1, figsize=(12, 2.5 * output_dim))
    if output_dim == 1:
        axes = [axes]

    time_steps = np.arange(horizon)

    for i in range(output_dim):
        axes[i].plot(time_steps, ground_truth[:, i], 'b-',
                    label='Ground Truth', linewidth=2, alpha=0.8)
        axes[i].plot(time_steps, predictions[:, i], 'r--',
                    label='Prediction', linewidth=2, alpha=0.8)
        axes[i].set_ylabel(state_labels[i])
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    plt.suptitle(f'Rollout Predictions (MSE: {rollout_results["mse"]:.6f})')
    plt.tight_layout()
    plt.show()


def plot_prediction_errors(rollout_results, state_labels=None):
    """
    Plot prediction errors over rollout horizon.

    Args:
        rollout_results: Dict from evaluate_rollout()
        state_labels: List of state dimension labels (optional)
    """
    errors = rollout_results['errors']
    horizon = errors.shape[0]
    output_dim = errors.shape[1]

    if state_labels is None:
        state_labels = [f'State {i}' for i in range(output_dim)]

    fig, axes = plt.subplots(output_dim, 1, figsize=(12, 2.5 * output_dim))
    if output_dim == 1:
        axes = [axes]

    time_steps = np.arange(horizon)

    for i in range(output_dim):
        axes[i].plot(time_steps, errors[:, i], 'k-', linewidth=1.5)
        axes[i].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[i].set_ylabel(f'{state_labels[i]} Error')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    plt.suptitle('Prediction Errors Over Rollout')
    plt.tight_layout()
    plt.show()


def plot_state_trajectories_3d(states, labels=None):
    """
    Plot 3D position trajectory.

    Args:
        states: [N, 3+] array with at least x, y, z positions
        labels: Dict with 'title', 'start_label', 'end_label'
    """
    if labels is None:
        labels = {
            'title': '3D Trajectory',
            'start_label': 'Start',
            'end_label': 'End'
        }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(states[:, 0], states[:, 1], states[:, 2],
            'b-', linewidth=2, alpha=0.7)

    # Mark start and end
    ax.scatter([states[0, 0]], [states[0, 1]], [states[0, 2]],
               c='green', s=100, marker='o', label=labels['start_label'])
    ax.scatter([states[-1, 0]], [states[-1, 1]], [states[-1, 2]],
               c='red', s=100, marker='x', label=labels['end_label'])

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(labels['title'])
    ax.legend()

    plt.show()


def plot_data_distribution(X, Y, feature_names=None, output_names=None):
    """
    Plot distributions of input features and output targets.

    Args:
        X: Input features [N, input_dim]
        Y: Output targets [N, output_dim]
        feature_names: List of input feature names (optional)
        output_names: List of output target names (optional)
    """
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    if feature_names is None:
        feature_names = [f'Input {i}' for i in range(input_dim)]
    if output_names is None:
        output_names = [f'Output {i}' for i in range(output_dim)]

    # Plot input distributions
    n_rows = int(np.ceil(input_dim / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if input_dim > 1 else [axes]

    for i in range(input_dim):
        # Use adaptive binning to handle columns with little variation
        data_range = X[:, i].max() - X[:, i].min()
        if data_range < 1e-10:
            # Nearly constant data - use fewer bins
            bins = 5
        else:
            # Use auto binning for varied data
            bins = 'auto'

        axes[i].hist(X[:, i], bins=bins, alpha=0.7, edgecolor='black')
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(input_dim, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Input Feature Distributions')
    plt.tight_layout()
    plt.show()

    # Plot output distributions
    n_rows = int(np.ceil(output_dim / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if output_dim > 1 else [axes]

    for i in range(output_dim):
        # Use adaptive binning to handle columns with little variation
        data_range = Y[:, i].max() - Y[:, i].min()
        if data_range < 1e-10:
            # Nearly constant data - use fewer bins
            bins = 5
        else:
            # Use auto binning for varied data
            bins = 'auto'

        axes[i].hist(Y[:, i], bins=bins, alpha=0.7, edgecolor='black', color='orange')
        axes[i].set_title(output_names[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(output_dim, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Output Target Distributions (Deltas)')
    plt.tight_layout()
    plt.show()


def plot_control_forces(controls, dt=1/80.0, labels=None):
    """
    Plot control forces over time.

    Args:
        controls: List or array of control inputs [N, 4]
        dt: Time step (default: 1/80s)
        labels: List of control labels (default: ['f1', 'f2', 'f3', 'f4'])
    """
    controls = np.array(controls)
    N = controls.shape[0]
    time = np.arange(N) * dt

    if labels is None:
        labels = ['f1', 'f2', 'f3', 'f4']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Individual forces
    for i in range(4):
        axes[0].plot(time, controls[:, i] * 1000, label=labels[i], linewidth=1.5)
    axes[0].set_ylabel('Force [mN]')
    axes[0].set_title('Per-Wing Forces')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Total force
    total_force = controls.sum(axis=1) * 1000
    axes[1].plot(time, total_force, 'k-', linewidth=2)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Total Force [mN]')
    axes[1].set_title('Total Thrust')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_dataset_summary(X, Y):
    """
    Print summary statistics for dataset.

    Args:
        X: Input features [N, input_dim]
        Y: Output targets [N, output_dim]
    """
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Input dimension:   {X.shape[1]}")
    print(f"Output dimension:  {Y.shape[1]}")
    print(f"\nInput statistics:")
    print(f"  Mean:  {X.mean(axis=0)}")
    print(f"  Std:   {X.std(axis=0)}")
    print(f"  Min:   {X.min(axis=0)}")
    print(f"  Max:   {X.max(axis=0)}")
    print(f"\nOutput statistics:")
    print(f"  Mean:  {Y.mean(axis=0)}")
    print(f"  Std:   {Y.std(axis=0)}")
    print(f"  Min:   {Y.min(axis=0)}")
    print(f"  Max:   {Y.max(axis=0)}")
    print(f"{'='*60}\n")


def print_model_summary(model):
    """
    Print model architecture summary.

    Args:
        model: PyTorch model
    """
    print(f"\n{'='*60}")
    print("Model Summary")
    print(f"{'='*60}")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")
