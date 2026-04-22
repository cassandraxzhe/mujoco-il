"""
Model Predictive Control (MPC) using Cross-Entropy Method (CEM) optimization.

Wing layout (matches hopper.xml):
    wing1: (+L, +L)   wing2: (+L, -L)
    wing3: (-L, +L)   wing4: (-L, -L)

Torque convention (τ = r × F, F vertical):
    Tx =  L * ( f1 - f2 + f3 - f4)
    Ty =  L * (-f1 - f2 + f3 + f4)
    Tz =  0  (symmetric layout, no yaw torque)
"""

import numpy as np
import torch
import mujoco


# ============================================================================
# DEFAULT MPC PARAMETERS
# ============================================================================

# These defaults are overridden when you pass explicit arguments,
# or can be updated by loading a config: from configs.mpc_config import load_config

FMAX_WING = 0.003       # Per-wing max force (N) — 3 mN
L         = 0.015       # Torso-center to wing, meters (15 mm)
DT        = 1 / 80.0    # Control timestep (80 Hz)
H         = 20          # MPC horizon steps (~0.25 s at 80 Hz)
POP       = 256         # CEM population size
ELITES    = 32          # CEM elite samples
ITERS     = 4           # CEM iterations per step
CTRL_DIM  = 4           # [f1, f2, f3, f4]

# Cost weights
Z_DES   = 0.08   # Desired hover height (m) — 8 cm is realistic for hopping
W_UP    = 6.0    # Attitude stability (roll/pitch penalty)
W_Z     = 5.0    # Height tracking
W_XY    = 2.0    # Lateral drift penalty
W_U     = 2e-4   # Control effort
W_OMEGA = 2.0    # Angular velocity damping

# Ground contact
GROUND_Z_THRESHOLD = 0.01   # Below 1 cm → considered grounded

# Normalization statistics — set via set_normalization_stats() after training
X_MEAN, X_STD = None, None
Y_MEAN, Y_STD = None, None


# ============================================================================
# NORMALIZATION
# ============================================================================

def set_normalization_stats(x_mean, x_std, y_mean, y_std):
    """
    Register training normalization statistics.

    Must be called before running any simulation that uses predict_next(),
    if the model was trained on normalized data.

    Args:
        x_mean: Input mean  [input_dim]
        x_std:  Input std   [input_dim]
        y_mean: Output mean [output_dim]
        y_std:  Output std  [output_dim]
    """
    global X_MEAN, X_STD, Y_MEAN, Y_STD
    X_MEAN, X_STD = x_mean, x_std
    Y_MEAN, Y_STD = y_mean, y_std


def normalize_in(X):
    """Normalize inputs using registered training statistics."""
    if X_MEAN is None:
        return X
    return (X - X_MEAN) / (X_STD + 1e-8)


def denormalize_out(Y):
    """Denormalize model outputs using registered training statistics."""
    if Y_MEAN is None:
        return Y
    return Y * (Y_STD + 1e-8) + Y_MEAN


# ============================================================================
# STATE-ACTION CONVERSION
# ============================================================================

def state_action_to_input(pos, eul, u_vec, L_arm=L):
    """
    Convert state + per-wing forces to NN input vector.

    Wing layout assumed (matches hopper.xml):
        f1=(+L,+L), f2=(+L,-L), f3=(-L,+L), f4=(-L,-L)

    Torques from τ = r × F (F vertical):
        Tx =  L*( f1 - f2 + f3 - f4)
        Ty =  L*(-f1 - f2 + f3 + f4)
        Tz =  0

    Args:
        pos:   [3] position (x, y, z) in meters
        eul:   [3] euler angles (roll, pitch, yaw) in radians
        u_vec: [4] per-wing forces (f1, f2, f3, f4) in Newtons
        L_arm: moment arm in meters

    Returns:
        inp: [14] vector [pos(3), eul(3), F(1), tau(3), signals(4)]
    """
    f1, f2, f3, f4 = u_vec

    F  =  f1 + f2 + f3 + f4
    Tx =  L_arm * ( f1 - f2 + f3 - f4)
    Ty =  L_arm * (-f1 - f2 + f3 + f4)
    Tz =  0.0

    return np.array([
        pos[0], pos[1], pos[2],
        eul[0], eul[1], eul[2],
        F,
        Tx, Ty, Tz,
        f1, f2, f3, f4,
    ], dtype=np.float32)


def _state_action_to_input_batch(pos, eul, U, L_arm=L):
    """
    Vectorized version of state_action_to_input for CEM rollouts.

    Args:
        pos: [N, 3] positions
        eul: [N, 3] euler angles
        U:   [N, 4] per-wing forces
        L_arm: moment arm in meters

    Returns:
        X: [N, 14] NN input batch
    """
    F  =  U.sum(axis=1, keepdims=True)                                  # [N,1]
    Tx = (L_arm * ( U[:,0] - U[:,1] + U[:,2] - U[:,3])).reshape(-1,1)  # [N,1]
    Ty = (L_arm * (-U[:,0] - U[:,1] + U[:,2] + U[:,3])).reshape(-1,1)  # [N,1]
    Tz =  np.zeros((len(U), 1), dtype=np.float32)

    return np.concatenate([pos, eul, F, Tx, Ty, Tz, U], axis=1).astype(np.float32)


# ============================================================================
# SINGLE-STEP DYNAMICS PREDICTION
# ============================================================================

def predict_next(model, pos, eul, u_vec, device='cpu'):
    """
    Predict next state using the learned dynamics model (single sample).

    Args:
        model:  Trained PyTorch model
        pos:    [3] current position
        eul:    [3] current euler angles
        u_vec:  [4] per-wing forces
        device: torch device string

    Returns:
        pos_next: [3] predicted next position
        eul_next: [3] predicted next euler angles
    """
    x_in = normalize_in(state_action_to_input(pos, eul, u_vec))
    x_t  = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        delta = denormalize_out(model(x_t).cpu().numpy()[0])  # [6]

    return pos + delta[:3], eul + delta[3:6]


# ============================================================================
# GROUND CONTACT DETECTION
# ============================================================================

def detect_ground_contact(mjmodel, mjdata,
                          z_threshold=GROUND_Z_THRESHOLD):
    """
    Detect whether the hopper is in contact with the ground.

    Uses both body height and MuJoCo's contact list. For a freejoint body,
    qpos[2] is the z coordinate of the root body.

    Args:
        mjmodel:     MuJoCo model
        mjdata:      MuJoCo data
        z_threshold: Height below which the robot is considered grounded (m)

    Returns:
        is_grounded: bool
        z_height:    float, current z position (m)
    """
    z_height    = float(mjdata.qpos[2])
    has_contact = mjdata.ncon > 0
    is_grounded = (z_height < z_threshold) or has_contact
    return is_grounded, z_height


# ============================================================================
# MPC COST FUNCTION
# ============================================================================

def mpc_cost_batch(traj_pos, traj_eul, traj_u, traj_omega=None,
                   z_des=Z_DES, w_up=W_UP, w_z=W_Z, w_u=W_U,
                   w_omega=W_OMEGA, w_xy=W_XY):
    """
    Vectorized MPC cost over a batch of trajectories.

    Args:
        traj_pos:   [N, H, 3] position trajectories
        traj_eul:   [N, H, 3] euler angle trajectories
        traj_u:     [N, H, 4] control trajectories
        traj_omega: [N, H, 3] angular velocity trajectories (optional)
        z_des:  desired height (m)
        w_up:   attitude weight
        w_z:    height weight
        w_u:    control effort weight
        w_omega: angular velocity weight
        w_xy:   lateral drift weight

    Returns:
        costs: [N] total cost per trajectory
    """
    roll  = traj_eul[:, :, 0]   # [N, H]
    pitch = traj_eul[:, :, 1]
    x     = traj_pos[:, :, 0]
    y     = traj_pos[:, :, 1]
    z     = traj_pos[:, :, 2]

    c_att  = w_up    * (roll**2 + pitch**2).sum(axis=1)      # [N]
    c_z    = w_z     * ((z - z_des)**2).sum(axis=1)          # [N]
    c_xy   = w_xy    * (x**2 + y**2).sum(axis=1)             # [N]
    c_u    = w_u     * (traj_u**2).sum(axis=(1, 2))          # [N]
    c_om   = (w_omega * (traj_omega**2).sum(axis=(1, 2))
              if traj_omega is not None else 0.0)

    return c_att + c_z + c_xy + c_u + c_om


def mpc_cost(traj_pos, traj_eul, traj_u, traj_omega=None,
             z_des=Z_DES, w_up=W_UP, w_z=W_Z, w_u=W_U,
             w_omega=W_OMEGA, w_xy=W_XY):
    """
    MPC cost for a single trajectory (scalar output). Kept for compatibility.

    Args:
        traj_pos:   [H, 3]
        traj_eul:   [H, 3]
        traj_u:     [H, 4]
        traj_omega: [H, 3] optional

    Returns:
        cost: scalar
    """
    omega_batch = traj_omega[None] if traj_omega is not None else None
    return float(mpc_cost_batch(
        traj_pos[None], traj_eul[None], traj_u[None], omega_batch,
        z_des=z_des, w_up=w_up, w_z=w_z, w_u=w_u,
        w_omega=w_omega, w_xy=w_xy,
    )[0])


# ============================================================================
# CEM OPTIMIZER  (vectorized rollout)
# ============================================================================

def cem_optimize(model, pos0, eul0, z_des=Z_DES, device='cpu',
                 h=H, pop=POP, elites=ELITES, iters=ITERS,
                 fmax=FMAX_WING, ctrl_dim=CTRL_DIM, verbose=False):
    """
    CEM optimization over a horizon of control sequences.

    The rollout is fully vectorized over the population — all `pop`
    candidate sequences are simulated simultaneously as a batch, avoiding
    the inner Python loop that was in the original implementation.

    The NN does not predict angular velocity, so omega is held constant
    at zero through the rollout. If angular velocity damping matters for
    your task, pass traj_omega from the simulator directly via mpc_step().

    Args:
        model:    Trained PyTorch dynamics model
        pos0:     [3] initial position
        eul0:     [3] initial euler angles
        z_des:    desired height (m)
        device:   torch device string
        h:        horizon length (steps)
        pop:      CEM population size
        elites:   number of elite samples kept each iteration
        iters:    number of CEM refinement iterations
        fmax:     maximum per-wing force (N)
        ctrl_dim: control dimension (always 4)
        verbose:  print per-iteration debug info

    Returns:
        u0: [4] best first control action
    """
    model.eval()

    # Initialise distribution — mean at quarter-hover, std at 25% of fmax
    mean = np.full((h, ctrl_dim), fmax * 0.25, dtype=np.float32)
    std  = np.full((h, ctrl_dim), fmax * 0.25, dtype=np.float32)

    best_u_seq = None
    best_cost  = np.inf

    for it in range(iters):
        # --- Sample [pop, h, 4] action sequences ---
        U = np.clip(
            np.random.normal(mean, std, size=(pop, h, ctrl_dim)).astype(np.float32),
            0.0, fmax,
        )

        # --- Vectorized rollout ---
        # pos_b, eul_b: [pop, 3] — current state for all samples
        pos_b = np.tile(pos0.astype(np.float32), (pop, 1))
        eul_b = np.tile(eul0.astype(np.float32), (pop, 1))

        traj_pos = np.zeros((pop, h, 3), dtype=np.float32)
        traj_eul = np.zeros((pop, h, 3), dtype=np.float32)

        with torch.no_grad():
            for t in range(h):
                u_t = U[:, t, :]    # [pop, 4]

                # Build NN inputs for all samples at once
                x_in = _state_action_to_input_batch(pos_b, eul_b, u_t)  # [pop, 14]
                x_in = normalize_in(x_in)

                delta = denormalize_out(
                    model(torch.tensor(x_in, dtype=torch.float32,
                                       device=device)).cpu().numpy()
                )   # [pop, 6]

                pos_b = pos_b + delta[:, :3]
                eul_b = eul_b + delta[:, 3:6]

                traj_pos[:, t, :] = pos_b
                traj_eul[:, t, :] = eul_b

        # --- Cost & elite selection ---
        costs     = mpc_cost_batch(traj_pos, traj_eul, U, z_des=z_des)
        elite_idx = np.argsort(costs)[:elites]
        elite_U   = U[elite_idx]   # [elites, h, 4]

        # --- Update distribution ---
        mean = elite_U.mean(axis=0)
        std  = elite_U.std(axis=0) + 1e-6

        if costs[elite_idx[0]] < best_cost:
            best_cost  = costs[elite_idx[0]]
            best_u_seq = elite_U[0].copy()

        if verbose:
            mean_f = best_u_seq[0].mean() * 1000
            print(f"  CEM iter {it}: cost={best_cost:.4f}  "
                  f"mean_f0={mean_f:.3f} mN")

    # Receding-horizon: return only the first action
    return best_u_seq[0]


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

def mpc_step(model, pos, eul, z_des=Z_DES, device='cpu', verbose=False):
    """
    Single MPC step — thin wrapper around cem_optimize().

    Args:
        model:   Trained dynamics model
        pos:     [3] current position
        eul:     [3] current euler angles
        z_des:   desired height (m)
        device:  torch device string
        verbose: debug output

    Returns:
        u: [4] per-wing forces
    """
    return cem_optimize(model, pos, eul, z_des=z_des,
                        device=device, verbose=verbose)