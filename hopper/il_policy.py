"""
Imitation-learning policy for the hopper.

A small MLP maps an observed state + target height to per-wing forces,
trained to clone a PD controller. This replaces the NN-dynamics + CEM
stack with a direct state->control mapping.

State layout (11-dim, translation-invariant):
    [x − x_des, y − y_des, z,
     vx, vy, vz,
     roll, pitch, omega_roll, omega_pitch,
     z_des]

The first two dims are position ERRORS relative to the horizontal
setpoint (x_des, y_des), not absolute world coordinates. This forces
translation invariance: the policy's behaviour depends only on how far
off it is from its commanded waypoint, not where in the world the
waypoint lives. For pure hovering, x_des = y_des = 0 (the default)
and the first two dims equal the absolute position — old pure-hover
demos are semantically identical under this convention. For forward
hopping or stair climbing, x_des (or y_des) is advanced over time and
the policy sees a persistent negative error, which drives pitch /
roll corrections that produce forward motion.

Output (4-dim): per-wing forces [f1, f2, f3, f4] in N.
"""

import numpy as np
import torch
import torch.nn as nn


IL_STATE_DIM = 11
IL_ACTION_DIM = 4
FMAX_WING_DEFAULT = 0.003


class ILPolicy(nn.Module):
    def __init__(self, input_dim=IL_STATE_DIM, output_dim=IL_ACTION_DIM,
                 hidden_dim=64, fmax=FMAX_WING_DEFAULT):
        super().__init__()
        self.fmax = float(fmax)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Raw output gated to [0, fmax] by a sigmoid — keeps actions feasible.
        return self.fmax * torch.sigmoid(self.net(x))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


class ILPolicyFTxTy(nn.Module):
    """
    Structured policy: predicts 3 physically meaningful scalars — total
    thrust F, body-x torque Tx, body-y torque Ty — and derives the four
    per-wing forces via the same analytical mixer used by the experts.

    This enforces symmetry by construction: a symmetric input state produces
    zero Tx and Ty, which the mixer turns into four identical wing forces.
    The direct 4-output `ILPolicy` cannot enforce this — any residual
    regression error injects asymmetry that seeds lateral drift and breaks
    the hopper's limit cycle.

    Output scales:
        F  ∈ [0, 4·fmax]           via sigmoid
        Tx ∈ ±2·L·fmax             via tanh  (physically reachable bound when
        Ty ∈ ±2·L·fmax             via tanh   F=2·fmax and antipodal pair saturated)
    """

    def __init__(self, input_dim=IL_STATE_DIM, hidden_dim=64,
                 fmax=FMAX_WING_DEFAULT, L=0.015):
        super().__init__()
        self.fmax = float(fmax)
        self.L = float(L)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        """Return (F, Tx, Ty) as a [..., 3] tensor with physical scaling."""
        raw = self.net(x)
        F = 4.0 * self.fmax * torch.sigmoid(raw[..., 0])
        Tx_bound = 2.0 * self.L * self.fmax
        Tx = Tx_bound * torch.tanh(raw[..., 1])
        Ty = Tx_bound * torch.tanh(raw[..., 2])
        return torch.stack([F, Tx, Ty], dim=-1)

    def wing_forces(self, x):
        """Run forward() + mixer → four per-wing forces clamped to [0, fmax]."""
        ftt = self.forward(x)
        F = ftt[..., 0]
        Tx = ftt[..., 1]
        Ty = ftt[..., 2]
        inv4L = 1.0 / (4.0 * self.L)
        f1 = F / 4.0 + (Tx - Ty) * inv4L
        f2 = F / 4.0 - (Tx + Ty) * inv4L
        f3 = F / 4.0 + (Tx + Ty) * inv4L
        f4 = F / 4.0 + (Ty - Tx) * inv4L
        return torch.stack([f1, f2, f3, f4], dim=-1).clamp(0.0, self.fmax)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


def wing_forces_to_ftxty(Y_wings, L=0.015):
    """Invert the mixer to get (F, Tx, Ty) from per-wing forces.

    Y_wings: [N, 4] or [4] array of per-wing forces (f1..f4).
    Returns: same-shape leading dims with 3 columns.
    """
    Y = np.asarray(Y_wings)
    f1, f2, f3, f4 = Y[..., 0], Y[..., 1], Y[..., 2], Y[..., 3]
    F = f1 + f2 + f3 + f4
    Tx = L * (f1 - f2 + f3 - f4)
    Ty = L * (-f1 - f2 + f3 + f4)
    return np.stack([F, Tx, Ty], axis=-1)


def extract_il_state(mjmodel, mjdata, z_des,
                     x_des=0.0, y_des=0.0,
                     body_name="hopper", **kwargs):
    """
    Build the 11-dim translation-invariant IL state vector.

    First two dimensions are horizontal position ERRORS (x − x_des, y − y_des);
    all other dims are body state. qvel layout for a freejoint is
    [vx, vy, vz (world), wx, wy, wz (body)]. Unknown kwargs (e.g. last_apex)
    are silently ignored for backwards compatibility.
    """
    from .simulation import get_body_pos_eul, body_name_to_id  # local to avoid cycle
    pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name=body_name)
    vx = float(mjdata.qvel[0])
    vy = float(mjdata.qvel[1])
    vz = float(mjdata.qvel[2])
    wr = float(mjdata.qvel[3])
    wp = float(mjdata.qvel[4])
    return np.array(
        [pos[0] - x_des, pos[1] - y_des, pos[2],
         vx, vy, vz,
         eul[0], eul[1], wr, wp, z_des],
        dtype=np.float32,
    )
