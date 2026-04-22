"""
Frame realignment utilities for orientation-invariant control.

These functions implement the frame realignment approach from the supplement,
allowing the controller to work robustly across different body orientations.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def normalize(v, eps=1e-9):
    """
    Normalize a vector.

    Args:
        v: Input vector
        eps: Small epsilon for numerical stability

    Returns:
        v_normalized: Normalized vector (or original if norm is too small)
    """
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


# ============================================================================
# FRAME REALIGNMENT
# ============================================================================

def build_R_match(R_body, R_mpc_1, eps=1e-6):
    """
    Construct R_match to realign body frame -> NN frame.

    This function builds an alignment matrix that maps the body frame to a
    canonical frame for the neural network, making control orientation-invariant.

    Args:
        R_body: [3, 3] body-to-world rotation matrix
        R_mpc_1: [3] desired reference x-axis from MPC (world frame)
        eps: Tolerance for degenerate cross products

    Returns:
        R_match: [3, 3] alignment matrix (body -> NN frame)
    """
    # Current z-axis (vertical, third column of R_body)
    R3_match = normalize(R_body[:, 2])

    # Desired x-axis direction
    R_mpc_1 = normalize(R_mpc_1)

    # New y-axis: cross product of z with desired x
    cross = np.cross(R3_match, R_mpc_1)

    # Handle near-parallel case (z-axis aligned with desired x-axis)
    if np.linalg.norm(cross) < eps:
        # Pick an arbitrary orthogonal axis as a safe default
        if abs(R3_match[2]) < 0.9:
            # z-axis not aligned with world z, use world z
            cross = np.cross(R3_match, np.array([0, 0, 1]))
        else:
            # z-axis aligned with world z, use world y
            cross = np.cross(R3_match, np.array([0, 1, 0]))

    R2_match = normalize(cross)

    # New x-axis: complete the orthonormal basis
    R1_match = normalize(np.cross(R2_match, R3_match))

    # Build rotation matrix from column vectors
    R_match = np.column_stack((R1_match, R2_match, R3_match))

    # Enforce right-handedness
    if np.linalg.det(R_match) < 0:
        R_match[:, 1] *= -1  # Flip y-axis

    return R_match


def body_to_nn(R_body, omega_body, R_match):
    """
    Transform body rotation and angular velocity to NN frame.

    Args:
        R_body: [3, 3] body rotation matrix
        omega_body: [3] angular velocity in body frame (rad/s)
        R_match: [3, 3] alignment matrix

    Returns:
        q_nn: [4] quaternion in NN frame (x, y, z, w)
        omega_nn: [3] angular velocity in NN frame (rad/s)
    """
    # Transform rotation
    R_nn = R_match @ R_body
    q_nn = R.from_matrix(R_nn).as_quat()  # (x, y, z, w) format

    # Transform angular velocity
    omega_nn = R_match @ omega_body

    return q_nn, omega_nn


def tau_nn_to_body(tau_nn, R_match):
    """
    Transform torque from NN frame back to body frame.

    Args:
        tau_nn: [3] torque in NN frame (Nm)
        R_match: [3, 3] alignment matrix

    Returns:
        tau_body: [3] torque in body frame (Nm)
    """
    return R_match.T @ tau_nn


# ============================================================================
# PER-WING FORCE ALLOCATION
# ============================================================================

# Geometry and limits
Lx = 0.015           # Moment arm about x (m)
Ly = 0.015           # Moment arm about y (m)
FMAX_WING = 0.0036   # Per-wing max thrust (N)

# Allocation matrix A' from supplement
# Wing layout: 1=(-x,-y), 2=(+x,-y), 3=(+x,+y), 4=(-x,+y)
Aprime = np.array([
    [-1.0, -1.0],   # Wing 1
    [ 1.0, -1.0],   # Wing 2
    [ 1.0,  1.0],   # Wing 3
    [-1.0,  1.0]    # Wing 4
], dtype=np.float64)


def allocate_per_wing_forces(F_cmd, Tx_cmd, Ty_cmd, Tx_ext=0.0, Ty_ext=0.0,
                             Lx_arm=Lx, Ly_arm=Ly, fmax=FMAX_WING):
    """
    Compute individual per-wing thrusts from total force + torques.

    Uses the allocation algorithm from the supplement to distribute forces
    across wings while respecting actuator limits.

    Args:
        F_cmd: Total commanded thrust (N)
        Tx_cmd: Commanded torque about x-axis (Nm)
        Ty_cmd: Commanded torque about y-axis (Nm)
        Tx_ext: External torque about x to compensate (Nm)
        Ty_ext: External torque about y to compensate (Nm)
        Lx_arm: Moment arm for x-axis (m)
        Ly_arm: Moment arm for y-axis (m)
        fmax: Maximum force per wing (N)

    Returns:
        f: [4] per-wing forces (N), non-negative, <= fmax
    """
    # 1) Compute required differential forces u'
    u_prime = 0.25 * np.array([
        (Tx_cmd - Tx_ext) / max(Lx_arm, 1e-12),
        (Ty_cmd - Ty_ext) / max(Ly_arm, 1e-12)
    ])

    # 2) Compute sigma = A' * u'
    sigma = Aprime @ u_prime  # [4]

    # 3) Initial equal force per actuator
    f_ini = F_cmd / 4.0

    # 4) Shift to avoid negative forces
    f_ini_prime = max(f_ini + np.min(sigma), 0.0)

    # 5) Compute final forces
    f = f_ini_prime + sigma

    # 6) Enforce per-wing bounds [0, fmax]
    # If any value > fmax, scale the whole vector down
    if np.max(f) > fmax:
        f *= fmax / np.max(f)

    # Final clipping
    f = np.clip(f, 0.0, fmax)

    return f


# ============================================================================
# MUJOCO ACCESSORS
# ============================================================================

def get_body_rotation_matrix(model, data, body_name="main"):
    """
    Safely extract 3×3 body rotation matrix from MuJoCo data.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of body

    Returns:
        R_body: [3, 3] rotation matrix
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # Handle different MuJoCo versions (flat vs. 2D array)
    xmat = np.asarray(data.xmat)
    if xmat.ndim == 1:  # Flat array
        R_body = xmat[bid*9 : bid*9 + 9].reshape(3, 3)
    else:  # 2D array
        R_body = xmat[bid].reshape(3, 3)

    # Safety check
    if not np.isfinite(R_body).all() or np.linalg.norm(R_body) < 1e-9:
        print(f"[WARN] Invalid R_body for '{body_name}': replacing with identity.")
        R_body = np.eye(3)

    return R_body


def get_body_omega(data):
    """
    Extract angular velocity from MuJoCo data.

    For a freejoint, qvel[3:6] are angular velocities (rad/s).

    Args:
        data: MuJoCo data

    Returns:
        omega: [3] angular velocity (rad/s)
    """
    return np.array(data.qvel[3:6])


def set_wing_forces_by_name(model, data, f_vec):
    """
    Assign per-wing thrusts to actuators by name.

    Supports multiple naming patterns: 'thrust1', 'thrust2', ... or 'f1', 'f2', ...

    Args:
        model: MuJoCo model
        data: MuJoCo data
        f_vec: [4] per-wing forces (N)
    """
    # Supported actuator naming patterns
    name_patterns = ["thrust{}", "f{}"]

    for i in range(4):
        assigned = False
        for pattern in name_patterns:
            name = pattern.format(i + 1)
            try:
                a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[a_id] = float(f_vec[i])
                assigned = True
                break
            except Exception:
                continue

        if not assigned:
            print(f"[WARN] Could not assign actuator for wing {i+1}")


# ============================================================================
# HARDWARE MAPPING (OPTIONAL)
# ============================================================================

# Placeholder per-wing force-to-voltage mapping
# Format: (a, b) for V = a*f + b
# Must be calibrated on real hardware
F2V_PARAMS = [(1e3, 0.0)] * 4


def f_to_voltage(f_vector):
    """
    Convert thrusts (N) to estimated voltage (V).

    Uses linear mapping V = a*f + b with parameters from F2V_PARAMS.
    Must be calibrated on actual hardware.

    Args:
        f_vector: [4] per-wing forces (N)

    Returns:
        V: [4] per-wing voltages (V)
    """
    return np.array([a*f + b for (a, b), f in zip(F2V_PARAMS, f_vector)])
