# control_utils.py
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

# -------------------------------------------------
# Frame realignment utilities
# -------------------------------------------------
def normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def build_R_match(R_body, R_mpc_1, eps=1e-6):
    """
    Construct R_match to realign body -> NN frame.
    Includes safeguards for degenerate cross products.
    """
    R3_match = normalize(R_body[:, 2])  # body z-axis
    R_mpc_1 = normalize(R_mpc_1)

    # cross product for new y-axis
    cross = np.cross(R3_match, R_mpc_1)

    # handle near-parallel case
    if np.linalg.norm(cross) < eps:
        # pick an arbitrary orthogonal axis (safe default)
        if abs(R3_match[2]) < 0.9:
            cross = np.cross(R3_match, np.array([0, 0, 1]))
        else:
            cross = np.cross(R3_match, np.array([0, 1, 0]))

    R2_match = normalize(cross)
    R1_match = normalize(np.cross(R2_match, R3_match))
    R_match = np.column_stack((R1_match, R2_match, R3_match))

    # enforce right-handedness
    if np.linalg.det(R_match) < 0:
        R_match[:, 1] *= -1  # flip y-axis
    return R_match


def body_to_nn(R_body, omega_body, R_match):
    """Transform body rotation and angular velocity to NN frame."""
    R_nn = R_match @ R_body
    q_nn = R.from_matrix(R_nn).as_quat()  # quaternion (x,y,z,w)
    omega_nn = R_match @ omega_body
    return q_nn, omega_nn

def tau_nn_to_body(tau_nn, R_match):
    """Transform torque from NN frame -> body frame."""
    return R_match.T @ tau_nn

# -------------------------------------------------
# Per-wing force allocation
# -------------------------------------------------
Lx, Ly = 0.015, 0.015
Fmax_wing = 0.0036

Aprime = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0]
], dtype=np.float64)

def allocate_per_wing_forces(F_cmd, Tx_cmd, Ty_cmd, Tx_ext=0.0, Ty_ext=0.0):
    """Compute individual per-wing thrusts from total force + torques."""
    u_prime = 0.25 * np.array([(Tx_cmd - Tx_ext)/max(Lx,1e-12),
                               (Ty_cmd - Ty_ext)/max(Ly,1e-12)])
    sigma = Aprime @ u_prime
    f_ini = F_cmd / 4.0
    f_ini_prime = max(f_ini + np.min(sigma), 0.0)
    f = f_ini_prime + sigma
    if np.max(f) > Fmax_wing:
        f *= Fmax_wing / np.max(f)
    return np.clip(f, 0.0, Fmax_wing)

# -------------------------------------------------
# Hardware mapping (optional)
# -------------------------------------------------
f2v_params = [(1e3, 0.0)] * 4
def f_to_voltage(f_vector):
    """Convert thrusts (N) to estimated voltage (V)."""
    return np.array([a*f + b for (a,b), f in zip(f2v_params, f_vector)])

# -------------------------------------------------
# MuJoCo accessors
# -------------------------------------------------
def get_body_rotation_matrix(model, data, body_name="main"):
    """Safely extract 3×3 body rotation matrix from MuJoCo data."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    # In some MuJoCo versions, data.xmat is (nbody, 9); in others, it's flat
    xmat = np.asarray(data.xmat)
    if xmat.ndim == 1:  # flat array
        R_body = xmat[bid*9 : bid*9 + 9].reshape(3, 3)
    else:
        R_body = xmat[bid].reshape(3, 3)

    if not np.isfinite(R_body).all() or np.linalg.norm(R_body) < 1e-9:
        print(f"[WARN] Invalid R_body for {body_name}: replacing with identity.")
        R_body = np.eye(3)

    return R_body



def get_body_omega(data):
    """Extract angular velocity from qvel (for freejoint)."""
    # For a freejoint, qvel[3:6] are angular velocities (rad/s)
    return np.array(data.qvel[3:6])

def set_wing_forces_by_name(model, data, f_vec):
    """
    Assigns per-wing thrusts to actuators by name.
    Works with either 'thrust1..4' or 'f1..4' naming.
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


# def set_wing_forces_by_name(data, f_vec):
#     """Apply per-wing thrusts to MuJoCo actuators."""
#     model = data.model
#     names = ['f1', 'f2', 'f3', 'f4']
#     for i, name in enumerate(names):
#         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
#         data.ctrl[idx] = float(f_vec[i])






### OLD
import numpy as np
# from scipy.spatial.transform import Rotation as R

# # Rmatch = [[],[],[]] # Alignment matrix
# # R = [[],[],[]] # Rotation matrix
# # tnn = [] # torque from nn
# # w = [] # angular velocity

# # # Constructing Rmatch
# # Rmatch[2] = R[2].copy() # copy third column of R into third column of Rmatch
# # Rmatch[1] = np.linalg.norm(np.cross(R[2], mpc(R[0])))
# # Rmatch[0] = np.linalg.norm(np.cross(Rmatch[1], Rmatch[2]))

# # qnn = Rotation.from_matrix(Rmatch * R).as_quat()
# # wnn = Rmatch * w
# # t = Rmatch.transpose * tnn



# # ---------------------------
# # Geometry / limits (tweak)
# # ---------------------------
# Lx = 0.015                    # moment arm about x (m)
# Ly = 0.015                    # moment arm about y (m) - can be same
# Fmax_wing = 0.0036            # per-wing max thrust (N) - change to your value

# # Allocation matrix A' from the supplement
# Aprime = np.array([
#     [-1.0, -1.0],
#     [ 1.0, -1.0],
#     [ 1.0,  1.0],
#     [-1.0,  1.0]
# ], dtype=np.float64)  # shape (4,2)

# # ---------------------------
# # Frame realign helpers
# # ---------------------------
# def normalize(v, eps=1e-9):
#     n = np.linalg.norm(v)
#     if n < eps:
#         return v
#     return v / n

# def build_R_match(R_body, R_mpc_1):
#     """
#     R_body: 3x3 body->world rotation matrix (columns are body axes in world frame)
#     R_mpc_1: 3-vector, desired reference x-axis from MPC (world frame)
#     Returns: R_match: 3x3 matrix mapping (body -> NN-frame) as described in supplement
#     """
#     # current z-axis (third column of R_body)
#     R3_match = R_body[:, 2]
#     # new y-axis: normalized cross(R3, R_mpc_x)
#     cross = np.cross(R3_match, R_mpc_1)
#     R2_match = normalize(cross)
#     # new x-axis is cross(y, z)
#     R1_match = normalize(np.cross(R2_match, R3_match))
#     R_match = np.column_stack((R1_match, R2_match, R3_match))
#     return R_match

# def body_to_nn(R_body, omega_body, R_match):
#     """transform body rotation & angular vel to NN frame"""
#     R_nn = R_match @ R_body      # R_nn = R_match * R (paper uses Rmatch R)
#     q_nn = R.from_matrix(R_nn).as_quat()  # (x,y,z,w)
#     omega_nn = R_match @ omega_body
#     return q_nn, omega_nn

# def tau_nn_to_body(tau_nn, R_match):
#     """transform torque output from NN frame -> body frame"""
#     return R_match.T @ tau_nn

# # ---------------------------
# # Thrust allocation
# # ---------------------------
# def allocate_per_wing_forces(F_cmd, Tx_cmd, Ty_cmd, Tx_ext=0.0, Ty_ext=0.0):
#     """
#     Inputs: all in body frame (N and N*m):
#       F_cmd : total commanded thrust (N)
#       Tx_cmd, Ty_cmd : commanded body torques (N*m) about x and y (body axes)
#       Tx_ext, Ty_ext : estimated external torques (N*m) to compensate (optional)
#     Returns:
#       f: array shape (4,) with per-wing forces (N), non-negative, <= Fmax_wing
#     """
#     # 1) compute required differential forces u' as in supplement
#     #    u' = 1/4 * [ (Tx_cmd - Tx_ext)/l_x, (Ty_cmd - Ty_ext)/l_y ]^T
#     u_prime = 0.25 * np.array([ (Tx_cmd - Tx_ext) / max(Lx, 1e-12),
#                                 (Ty_cmd - Ty_ext) / max(Ly, 1e-12) ])

#     # 2) sigma = A' * u'
#     sigma = Aprime @ u_prime   # shape (4,)

#     # 3) initial equal force per actuator
#     f_ini = F_cmd / 4.0

#     # 4) shift so that no f is negative (f'_ini = max(f_ini + min_i sigma_i, 0))
#     f_ini_prime = max(f_ini + np.min(sigma), 0.0)

#     # 5) final forces
#     f = f_ini_prime + sigma    # may have negatives clipped by f_ini_prime above

#     # 6) Enforce per-wing bounds: [0, Fmax_wing]
#     # If any value > Fmax_wing, scale the whole vector down so the maximum equals Fmax_wing.
#     # This preserves torque ratios approximatively while enforcing actuator limits.
#     if np.max(f) > Fmax_wing:
#         scale = Fmax_wing / np.max(f)
#         f = f * scale

#     # Finally clip numerically
#     f = np.clip(f, 0.0, Fmax_wing)
#     return f

# # ---------------------------
# # Optional: empirical f -> voltage mapping
# # ---------------------------
# # placeholder per-wing mapping functions (replace with fitted polynomials or lookup)
# # Example: simple linear mapping (V = a * f + b), parameters must be calibrated on real hardware.
# f2v_params = [
#     (1e3, 0.0),  # wing1: (a,b) -- example numbers; change after calibration
#     (1e3, 0.0),
#     (1e3, 0.0),
#     (1e3, 0.0)
# ]

# def f_to_voltage(f_vector):
#     """Return per-wing voltage amplitudes from force vector f_vector (N)."""
#     V = np.zeros(4)
#     for i in range(4):
#         a,b = f2v_params[i]
#         V[i] = a * f_vector[i] + b
#     return V