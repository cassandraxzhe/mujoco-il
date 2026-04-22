"""
Model Predictive Control (MPC) using Cross-Entropy Method (CEM) optimization.
"""

import numpy as np
import torch


# ============================================================================
# DEFAULT MPC PARAMETERS
# ============================================================================

# These can be overridden by importing from configs/mpc_config.py
FMAX_WING = 0.003       # per-wing max force (N) - 3 mN
L = 0.015               # 15 mm (m), distance torso center to wing
DT = 1/80.0             # control step (80 Hz)
H = 20                  # horizon steps for MPC (~0.25s at 80Hz)
POP = 256               # CEM population size
ELITES = 32             # CEM elite samples
ITERS = 4               # CEM iterations
CTRL_DIM = 4            # f1, f2, f3, f4 (per-wing forces)

# Cost function weights
Z_DES = 0.20            # desired height (m)
W_UP = 6.0              # attitude stability (roll/pitch penalty)
W_Z = 5.0               # height tracking
W_XY = 2.0              # lateral drift penalty
W_U = 2e-4              # control effort
W_OMEGA = 2.0           # angular velocity damping

# Optional normalization statistics (set after training if used)
X_MEAN, X_STD = None, None
Y_MEAN, Y_STD = None, None


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_in(X):
    """
    Normalize input features using training statistics.

    Args:
        X: Input features [batch, input_dim] or [input_dim]

    Returns:
        X_norm: Normalized inputs
    """
    if X_MEAN is None:
        return X
    return (X - X_MEAN) / (X_STD + 1e-8)


def denormalize_out(Y):
    """
    Denormalize model outputs using training statistics.

    Args:
        Y: Model outputs [batch, output_dim] or [output_dim]

    Returns:
        Y_denorm: Denormalized outputs
    """
    if Y_MEAN is None:
        return Y
    return Y * (Y_STD + 1e-8) + Y_MEAN


def set_normalization_stats(x_mean, x_std, y_mean, y_std):
    """
    Set normalization statistics from training data.

    Args:
        x_mean: Input mean [input_dim]
        x_std: Input std [input_dim]
        y_mean: Output mean [output_dim]
        y_std: Output std [output_dim]
    """
    global X_MEAN, X_STD, Y_MEAN, Y_STD
    X_MEAN = x_mean
    X_STD = x_std
    Y_MEAN = y_mean
    Y_STD = y_std


# ============================================================================
# STATE-ACTION TO INPUT CONVERSION
# ============================================================================

def state_action_to_input(pos, eul, u_vec, L_arm=L):
    """
    Convert state + action to neural network input format.

    Computes total thrust and torques from per-wing forces.

    Args:
        pos: Position [x, y, z] in meters
        eul: Euler angles [roll, pitch, yaw] in radians
        u_vec: Per-wing forces [f1, f2, f3, f4] in Newtons
        L_arm: Arm length (distance from torso center to wing) in meters

    Returns:
        inp: [14] input vector [pos(3), eul(3), F(1), tau(3), signals(4)]
    """
    f1, f2, f3, f4 = u_vec

    # Total thrust
    F = f1 + f2 + f3 + f4

    # Torques (assuming square wing layout)
    # Wing positions: f1=(-L,-L), f2=(L,-L), f3=(L,L), f4=(-L,L)
    Tx = -L_arm * f1 + L_arm * f2 + L_arm * f3 - L_arm * f4
    Ty = -L_arm * f1 - L_arm * f2 + L_arm * f3 + L_arm * f4
    Tz = 0.0  # No yaw torque from symmetric forces

    signals = np.array([f1, f2, f3, f4], dtype=np.float32)

    # Concatenate: [pos(3), eul(3), thrust(1), torque(3), signals(4)] = 14D
    inp = np.hstack([pos, eul, [F], [Tx, Ty, Tz], signals]).astype(np.float32)
    return inp


# ============================================================================
# DYNAMICS PREDICTION
# ============================================================================

def predict_next(model, pos, eul, u_vec, device='cpu'):
    """
    Predict next state using learned dynamics model.

    Args:
        model: Trained PyTorch dynamics model
        pos: Current position [3]
        eul: Current euler angles [3]
        u_vec: Control input (per-wing forces) [4]
        device: PyTorch device

    Returns:
        pos_next: Next position [3]
        eul_next: Next euler angles [3]
    """
    # Convert to model input format
    x_in = state_action_to_input(pos, eul, u_vec)

    # Normalize if statistics available
    x_in = normalize_in(x_in)

    # Run model
    x_t = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        y = model(x_t).cpu().numpy()[0]  # [6] predicted delta

    # Denormalize output
    y = denormalize_out(y)

    # Extract next state
    pos_next = pos + y[:3]   # Add delta to current position
    eul_next = eul + y[3:6]  # Add delta to current euler

    return pos_next, eul_next


# ============================================================================
# MPC COST FUNCTION
# ============================================================================

def mpc_cost(traj_pos, traj_eul, traj_u, traj_omega=None,
             z_des=Z_DES, w_up=W_UP, w_z=W_Z, w_u=W_U,
             w_omega=W_OMEGA, w_xy=W_XY):
    """
    MPC cost function for hopper control.

    Penalizes:
    - Attitude error (roll/pitch)
    - Height tracking error
    - Lateral drift (x/y position)
    - Control effort
    - Angular velocity (optional)

    Args:
        traj_pos: Position trajectory [H, 3]
        traj_eul: Euler angle trajectory [H, 3] (roll, pitch, yaw)
        traj_u: Control trajectory [H, 4]
        traj_omega: Angular velocity trajectory [H, 3] (optional)
        z_des: Desired height in meters
        w_up: Attitude weight
        w_z: Height tracking weight
        w_u: Control effort weight
        w_omega: Angular velocity damping weight
        w_xy: Lateral position weight

    Returns:
        cost: Scalar total cost
    """
    roll, pitch, yaw = traj_eul.T  # [H]
    x, y, z = traj_pos.T

    # Cost components
    c_up = w_up * (roll**2 + pitch**2)              # Attitude stability
    c_z = w_z * ((z - z_des)**2)                    # Height tracking
    c_xy = w_xy * (x**2 + y**2)                     # Lateral drift
    c_u = w_u * np.sum(traj_u**2, axis=1)           # Control effort

    # Angular velocity damping (optional)
    c_om = 0.0
    if traj_omega is not None:
        c_om = w_omega * np.sum(traj_omega**2, axis=1)

    # Total cost
    total_cost = np.sum(c_up + c_z + c_xy + c_u + c_om)
    return total_cost


# ============================================================================
# CEM OPTIMIZER
# ============================================================================

def cem_optimize(model, pos0, eul0, z_des=Z_DES, device='cpu',
                 h=H, pop=POP, elites=ELITES, iters=ITERS,
                 fmax=FMAX_WING, ctrl_dim=CTRL_DIM, verbose=False):
    """
    Cross-Entropy Method (CEM) optimization for MPC.

    Optimizes a sequence of control inputs to minimize MPC cost.

    Args:
        model: Trained dynamics model (PyTorch)
        pos0: Initial position [3]
        eul0: Initial euler angles [3]
        z_des: Desired height for cost function
        device: PyTorch device
        h: Horizon length
        pop: Population size
        elites: Number of elite samples
        iters: Number of CEM iterations
        fmax: Maximum per-wing force
        ctrl_dim: Control dimension (4 for per-wing forces)
        verbose: Print debug info

    Returns:
        u0: Best first control action [ctrl_dim]
    """
    # Initialize distribution
    mean = np.full((h, ctrl_dim), fmax * 0.25, dtype=np.float32)
    std = np.full((h, ctrl_dim), fmax * 0.25, dtype=np.float32)

    best_u_seq = None
    best_cost = np.inf

    for iter_idx in range(iters):
        # Sample action sequences from distribution
        U = np.random.normal(loc=mean, scale=std, size=(pop, h, ctrl_dim))
        U = U.astype(np.float32)
        U = np.clip(U, 0.0, fmax)

        # Evaluate each candidate sequence
        costs = np.zeros(pop, dtype=np.float64)
        for i in range(pop):
            pos = pos0.copy()
            eul = eul0.copy()
            traj_pos = []
            traj_eul = []

            # Rollout trajectory
            for t in range(h):
                u_t = U[i, t]
                pos, eul = predict_next(model, pos, eul, u_t, device=device)
                traj_pos.append(pos)
                traj_eul.append(eul)

            traj_pos = np.asarray(traj_pos)
            traj_eul = np.asarray(traj_eul)

            # Compute cost
            costs[i] = mpc_cost(traj_pos, traj_eul, U[i], z_des=z_des)

        # Select elites
        elite_idx = np.argsort(costs)[:elites]
        elite_U = U[elite_idx]

        # Update distribution
        mean = elite_U.mean(axis=0)
        std = elite_U.std(axis=0) + 1e-6  # Add small epsilon for stability

        # Track best sequence
        if costs[elite_idx[0]] < best_cost:
            best_cost = costs[elite_idx[0]]
            best_u_seq = elite_U[0].copy()

        if verbose:
            print(f"CEM iter {iter_idx}: best_cost={best_cost:.6f}")

    # Return first action (receding horizon)
    return best_u_seq[0]


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def mpc_step(model, pos, eul, z_des=Z_DES, device='cpu', verbose=False):
    """
    Single MPC step: run CEM optimization and return control action.

    Args:
        model: Trained dynamics model
        pos: Current position [3]
        eul: Current euler angles [3]
        z_des: Desired height
        device: PyTorch device
        verbose: Print debug info

    Returns:
        u: Control action [4] per-wing forces
    """
    return cem_optimize(model, pos, eul, z_des=z_des,
                       device=device, verbose=verbose)
