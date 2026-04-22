import numpy as np
import torch

# --------------------------
# TODO Helpers I need to edit
# --------------------------
FMAX_WING = 2.0e-5      # <-- per-wing max (N); set from your XML/mentor
L = 0.015               # 15 mm (m), distance torso center to wing
DT = 1/80.0             # control step (match your downsampled data)
H = 20                  # horizon steps for MPC (e.g., 20 @ 80Hz ≈ 0.25s)
POP = 256               # CEM population
ELITES = 32             # CEM elites
ITERS = 4               # CEM iterations
CTRL_DIM = 4            # f1..f4
DEVICE = "cpu"

# Optional input/output normalization (set these if you used them in training)
X_mean, X_std = None, None
Y_mean, Y_std = None, None

def normalize_in(X):
    if X_mean is None: return X
    return (X - X_mean) / (X_std + 1e-8)

def denormalize_out(Y):
    if Y_mean is None: return Y
    return Y * (Y_std + 1e-8) + Y_mean

# Build 14-D input for your MLP from state and action
def state_action_to_input(pos, eul, u_vec):
    # u = [f1..f4]
    f1,f2,f3,f4 = u_vec
    F  = f1 + f2 + f3 + f4
    Tx = -L*f1 + L*f2 + L*f3 - L*f4
    Ty = -L*f1 - L*f2 + L*f3 + L*f4
    # If your training used a 3-D torque, set Tz=0 (or derive appropriately)
    Tz = 0.0
    # Signals: if your 'rst_driving_signals' was the same as per-wing forces, use u directly.
    signals = np.array([f1,f2,f3,f4], dtype=np.float32)
    inp = np.hstack([pos, eul, [F], [Tx,Ty,Tz], signals]).astype(np.float32)
    return inp

# One learned dynamics step: x_{t+1} = f_hat(x_t, u_t)
def predict_next(model, pos, eul, u_vec):
    x_in = state_action_to_input(pos, eul, u_vec)
    x_in = normalize_in(x_in)
    x_t = torch.tensor(x_in, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        y = model(x_t).cpu().numpy()[0]   # 6-D next state
    y = denormalize_out(y)
    pos_next = y[:3]
    eul_next = y[3:6]
    return pos_next, eul_next

# Cost: track upright & desired COM height with small control penalty
def mpc_cost(traj_pos, traj_eul, traj_u, z_des=0.20, w_up=3.0, w_z=1.0, w_u=1e-3):
    """
    traj_pos: [H,3], traj_eul: [H,3], traj_u: [H,4]
    Cost encourages: near-zero roll/pitch, target height, small thrust.
    """
    roll, pitch = traj_eul[:,0], traj_eul[:,1]
    z = traj_pos[:,2]
    c_up = w_up * (roll**2 + pitch**2)
    c_z  = w_z  * ((z - z_des)**2)
    c_u  = w_u  * np.sum(traj_u**2, axis=1)
    return np.sum(c_up + c_z + c_u)

# CEM optimizer over an action *sequence* U = [u_0..u_{H-1}], u_t in [0,FMAX_WING]^4
def cem_optimize(model, pos0, eul0):
    mean = np.full((H, CTRL_DIM), FMAX_WING*0.25, dtype=np.float32)   # start modest
    std  = np.full((H, CTRL_DIM), FMAX_WING*0.25, dtype=np.float32)

    best_u_seq = None
    best_cost = np.inf

    for _ in range(ITERS):
        # sample
        U = np.random.normal(loc=mean, scale=std, size=(POP, H, CTRL_DIM)).astype(np.float32)
        U = np.clip(U, 0.0, FMAX_WING)

        # evaluate
        costs = np.zeros(POP, dtype=np.float64)
        for i in range(POP):
            pos = pos0.copy()
            eul = eul0.copy()
            traj_pos = []
            traj_eul = []
            for t in range(H):
                u_t = U[i, t]
                pos, eul = predict_next(model, pos, eul, u_t)
                traj_pos.append(pos)
                traj_eul.append(eul)
            traj_pos = np.asarray(traj_pos)
            traj_eul = np.asarray(traj_eul)
            costs[i] = mpc_cost(traj_pos, traj_eul, U[i])

        # elites
        elite_idx = np.argsort(costs)[:ELITES]
        elites = U[elite_idx]
        mean = elites.mean(axis=0)
        std  = elites.std(axis=0) + 1e-6

        if costs[elite_idx[0]] < best_cost:
            best_cost = costs[elite_idx[0]]
            best_u_seq = elites[0].copy()

    return best_u_seq[0]  # return only first control (receding horizon)

# --------------------------
# Tying into MuJoCo loop
# --------------------------
def mj_get_state(env):
    """
    dm_control style. If you use mujoco-py or mujoco-python, adapt to that API.
    Must return pos(3) and eulXYZ(3) consistent with your training.
    """
    # Example using named accessors (adjust to your model!)
    p = env.physics.named.data.xpos['torso']       # world COM or body position
    # If you trained on a specific position signal, read that instead.
    pos = np.array([p[0], p[1], p[2]], dtype=np.float32)

    # Euler XYZ from rotation matrix or quat
    R = env.physics.named.data.xmat['torso'].reshape(3,3)
    # naive XYZ extraction (ensure consistency with how you built rst_Eul_XYZ!)
    pitch = np.arcsin(-R[2,0])
    roll  = np.arctan2(R[2,1], R[2,2])
    yaw   = np.arctan2(R[1,0], R[0,0])
    eul = np.array([roll, pitch, yaw], dtype=np.float32)
    return pos, eul

def mj_apply_u(env, u_vec):
    """
    Write the chosen per-wing thrusts into MuJoCo controls.
    Names must match your XML (f1..f4 actuators).
    """
    env.physics.named.data.ctrl['f1'] = u_vec[0]
    env.physics.named.data.ctrl['f2'] = u_vec[1]
    env.physics.named.data.ctrl['f3'] = u_vec[2]
    env.physics.named.data.ctrl['f4'] = u_vec[3]

def mpc_control_step(env, model):
    pos, eul = mj_get_state(env)
    u0 = cem_optimize(model, pos, eul)
    mj_apply_u(env, u0)
    # advance MuJoCo by one control interval (match DT)
    # If your env uses smaller physics dt, step multiple times to match DT.
    substeps = max(1, int(DT / env.physics.model.opt.timestep + 1e-9))
    for _ in range(substeps):
        env.step()
    return u0
