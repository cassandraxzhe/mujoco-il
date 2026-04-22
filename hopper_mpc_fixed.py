"""
Fixed MPC control for sustained hopping behavior.

Key fixes:
1. Lower z_des from 0.20m to 0.08m for realistic hopping height
2. Remove/reduce hover bias - let CEM control full force range
3. Add ground contact detection
4. Proper force scaling: CEM optimizes in realistic force range
5. Add upper bounds on applied forces
"""

import numpy as np
import mujoco
import torch

# ============================================================================
# FIXED PARAMETERS
# ============================================================================

# MPC parameters - adjusted for hopping
FMAX_WING = 0.003           # Realistic max force per wing (3 mN) - was 2e-5
L = 0.015                   # 15 mm, distance torso center to wing
DT = 1/80.0                 # control step (80 Hz)
H = 20                      # MPC horizon
N_SAMPLES = 256             # CEM population
N_ELITE = 32                # CEM elite samples
N_ITER = 4                  # CEM iterations

# Mass and gravity
MASS = 0.00095 + 0.00020    # 1.15 mg total
G = 9.81
F_HOVER = MASS * G          # ~0.01128 N total hover force

# Cost function weights - adjusted for hopping
Z_DES_HOPPING = 0.08        # Target 8 cm height (realistic for hopping)
Z_DES_GROUND = 0.02         # Near-ground target when detecting contact
W_UP = 6.0                  # Attitude stability (roll/pitch)
W_Z = 5.0                   # Height tracking
W_XY = 2.0                  # Lateral drift
W_U = 2e-4                  # Control effort
W_OMEGA = 2.0               # Angular velocity damping

# Ground contact threshold
GROUND_HEIGHT_THRESHOLD = 0.01  # 1 cm - considered "on ground"


# ============================================================================
# FIXED MPC COST FUNCTION
# ============================================================================

def mpc_cost_fixed(traj_pos, traj_eul, traj_u, traj_omega, z_des=Z_DES_HOPPING,
                   w_up=W_UP, w_z=W_Z, w_u=W_U, w_omega=W_OMEGA, w_xy=W_XY):
    """
    Fixed cost function with proper weighting for hopping behavior.

    Args:
        traj_pos: [N, H, 3] position trajectories
        traj_eul: [N, H, 3] euler angle trajectories (roll, pitch, yaw)
        traj_u: [N, H, 4] control input trajectories (per-wing forces)
        traj_omega: [N, H, 3] angular velocity trajectories
        z_des: desired height

    Returns:
        costs: [N] total cost per trajectory
    """
    N = traj_pos.shape[0]
    H = traj_pos.shape[1]

    # Extract components
    x = traj_pos[:, :, 0]  # [N, H]
    y = traj_pos[:, :, 1]
    z = traj_pos[:, :, 2]

    roll = traj_eul[:, :, 0]
    pitch = traj_eul[:, :, 1]
    # yaw = traj_eul[:, :, 2]  # Not penalized

    # Cost components
    cost_attitude = w_up * (roll**2 + pitch**2).sum(axis=1)  # [N]
    cost_height = w_z * ((z - z_des)**2).sum(axis=1)          # [N]
    cost_lateral = w_xy * (x**2 + y**2).sum(axis=1)          # [N]
    cost_control = w_u * (traj_u**2).sum(axis=(1, 2))        # [N]
    cost_angvel = w_omega * (traj_omega**2).sum(axis=(1, 2)) # [N]

    total_cost = cost_attitude + cost_height + cost_lateral + cost_control + cost_angvel
    return total_cost


# ============================================================================
# GROUND CONTACT DETECTION
# ============================================================================

def detect_ground_contact(model, data):
    """
    Detect if hopper is in contact with ground.

    Returns:
        is_grounded: bool, True if on ground
        z_height: float, current height above ground
    """
    # Get body position (assuming first body is the main hopper body)
    z_height = data.qpos[2]  # z position

    # Check contact forces
    has_contact = data.ncon > 0

    # Consider grounded if below threshold OR has ground contact
    is_grounded = (z_height < GROUND_HEIGHT_THRESHOLD) or has_contact

    return is_grounded, z_height


# ============================================================================
# FIXED CEM OPTIMIZATION
# ============================================================================

def cem_optimize_fixed(net, pos0, eul0, omega0, z_des=Z_DES_HOPPING,
                       device='cpu', verbose=False):
    """
    Fixed CEM optimization with proper force scaling.

    Key fix: Optimize in realistic force range [0, FMAX_WING] instead of [0, 2e-5].
    No hover bias added here - CEM controls full force range.

    Args:
        net: trained dynamics model
        pos0: [3] initial position
        eul0: [3] initial euler angles
        omega0: [3] initial angular velocity
        z_des: desired height for cost function

    Returns:
        best_u: [4] best per-wing forces for first step
    """
    net.eval()

    # Initialize mean and std for CEM
    # Mean starts at hover force per wing
    mean = np.ones(4 * H) * (F_HOVER / 4.0)  # [H*4]
    std = np.ones(4 * H) * (FMAX_WING * 0.3)  # [H*4], 30% of max

    for iter_idx in range(N_ITER):
        # Sample action sequences
        U = np.random.normal(mean, std, size=(N_SAMPLES, 4 * H))  # [N, H*4]
        U = np.clip(U, 0.0, FMAX_WING)  # Clip to realistic bounds
        U_reshaped = U.reshape(N_SAMPLES, H, 4)  # [N, H, 4]

        # Rollout trajectories using learned model
        traj_pos = np.zeros((N_SAMPLES, H, 3))
        traj_eul = np.zeros((N_SAMPLES, H, 3))
        traj_omega = np.zeros((N_SAMPLES, H, 3))

        # Initial state for all samples
        pos_curr = np.tile(pos0, (N_SAMPLES, 1))  # [N, 3]
        eul_curr = np.tile(eul0, (N_SAMPLES, 1))  # [N, 3]
        omega_curr = np.tile(omega0, (N_SAMPLES, 1))  # [N, 3]

        # Rollout horizon
        with torch.no_grad():
            for t in range(H):
                u_t = U_reshaped[:, t, :]  # [N, 4]

                # Compute torques from forces (assuming square layout)
                tau_x = L * (u_t[:, 2] - u_t[:, 0])  # [N]
                tau_y = L * (u_t[:, 3] - u_t[:, 1])  # [N]
                tau_z = np.zeros(N_SAMPLES)  # No yaw torque from symmetric forces
                tau = np.stack([tau_x, tau_y, tau_z], axis=1)  # [N, 3]

                # Neural network input: [pos, eul, thrust, torque, signals]
                thrust_total = u_t.sum(axis=1, keepdims=True)  # [N, 1]

                x_in = np.concatenate([
                    pos_curr,      # [N, 3]
                    eul_curr,      # [N, 3]
                    thrust_total,  # [N, 1]
                    tau,           # [N, 3]
                    u_t            # [N, 4]
                ], axis=1)  # [N, 14]

                # Predict next state
                x_in_tensor = torch.from_numpy(x_in).float().to(device)
                delta = net(x_in_tensor).cpu().numpy()  # [N, 6]

                # Update state
                pos_next = pos_curr + delta[:, :3]
                eul_next = eul_curr + delta[:, 3:6]

                # Store trajectory
                traj_pos[:, t, :] = pos_next
                traj_eul[:, t, :] = eul_next
                traj_omega[:, t, :] = omega_curr  # Use current omega as approximation

                # Update for next step
                pos_curr = pos_next
                eul_curr = eul_next

        # Compute costs
        costs = mpc_cost_fixed(traj_pos, traj_eul, U_reshaped, traj_omega, z_des=z_des)

        # Select elites
        elite_idxs = np.argsort(costs)[:N_ELITE]
        elite_U = U[elite_idxs]  # [N_ELITE, H*4]

        # Update distribution
        mean = elite_U.mean(axis=0)
        std = elite_U.std(axis=0) + 1e-6  # Add small epsilon

        if verbose and iter_idx == N_ITER - 1:
            print(f"CEM iter {iter_idx}: best_cost={costs[elite_idxs[0]]:.6f}, "
                  f"mean_force={mean[:4].mean():.6f}N")

    # Return first action of best sequence
    best_idx = elite_idxs[0]
    best_u = U_reshaped[best_idx, 0, :]  # [4]

    return best_u


# ============================================================================
# FIXED MPC CONTROL STEP WITH GROUND DETECTION
# ============================================================================

def mpc_control_step_fixed(model, data, net, R_MPC_1=None,
                           apply_forces=True, verbose=False):
    """
    Fixed MPC control step with:
    - Ground contact detection
    - Adaptive z_des based on contact state
    - Proper force scaling (no excessive hover bias)
    - Force clipping with upper bounds

    Args:
        model: MuJoCo model
        data: MuJoCo data
        net: trained dynamics network
        R_MPC_1: optional frame alignment rotation matrix
        apply_forces: if True, apply forces to simulation

    Returns:
        f_vec: [4] applied wing forces
        tau_body: [3] body torques
        is_grounded: bool, ground contact status
    """
    # 1) Detect ground contact
    is_grounded, z_height = detect_ground_contact(model, data)

    # 2) Adaptive z_des based on contact state
    if is_grounded:
        # On ground: target low height, reduce forces
        z_des = Z_DES_GROUND
        force_scale = 0.5  # Reduce forces when on ground
    else:
        # Airborne: target hopping height
        z_des = Z_DES_HOPPING
        force_scale = 1.0

    # 3) Get current state
    pos = data.qpos[:3].copy()  # [x, y, z]

    # Get rotation matrix and euler angles
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    R_body = data.xmat[body_id].reshape(3, 3).copy()

    # Convert to euler angles (ZYX convention)
    eul = np.array([
        np.arctan2(R_body[2, 1], R_body[2, 2]),  # roll
        np.arctan2(-R_body[2, 0], np.sqrt(R_body[2, 1]**2 + R_body[2, 2]**2)),  # pitch
        np.arctan2(R_body[1, 0], R_body[0, 0])   # yaw
    ])

    # Get angular velocity
    omega_body = data.qvel[3:6].copy()  # [3] angular velocity in body frame

    # 4) Run CEM optimization
    f_vec_raw = cem_optimize_fixed(net, pos, eul, omega_body, z_des=z_des,
                                   verbose=verbose)

    # 5) Scale forces based on ground contact
    f_vec = f_vec_raw * force_scale

    # 6) Apply upper and lower bounds with safety margin
    f_vec = np.clip(f_vec, 0.0, FMAX_WING)

    # 7) Compute torques
    tau_x = L * (f_vec[2] - f_vec[0])
    tau_y = L * (f_vec[3] - f_vec[1])
    tau_z = 0.0
    tau_body = np.array([tau_x, tau_y, tau_z])

    # 8) Apply forces to simulation
    if apply_forces:
        # Wing forces (assuming actuators 0-3 are wings)
        for i in range(4):
            data.ctrl[i] = f_vec[i]

    if verbose:
        print(f"z={z_height:.4f}m, grounded={is_grounded}, z_des={z_des:.4f}m, "
              f"forces={f_vec.mean():.6f}N")

    return f_vec, tau_body, is_grounded


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Fixed MPC parameters:")
    print(f"  FMAX_WING: {FMAX_WING*1000:.3f} mN (was 0.02 mN)")
    print(f"  Z_DES_HOPPING: {Z_DES_HOPPING*100:.1f} cm (was 20 cm)")
    print(f"  Z_DES_GROUND: {Z_DES_GROUND*100:.1f} cm")
    print(f"  F_HOVER: {F_HOVER*1000:.3f} mN")
    print(f"  F_HOVER/4: {(F_HOVER/4)*1000:.3f} mN per wing")
    print(f"\nKey fixes:")
    print("  ✓ CEM optimizes in realistic force range [0, 3 mN]")
    print("  ✓ Removed excessive hover bias addition")
    print("  ✓ Added ground contact detection")
    print("  ✓ Adaptive z_des based on contact state")
    print("  ✓ Force scaling when on ground")
    print("  ✓ Proper upper bounds on applied forces")
