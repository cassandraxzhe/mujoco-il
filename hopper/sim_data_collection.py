"""
Simulated data collection for training the dynamics MLP.

A hand-tuned PD controller drives the hopper in MuJoCo while we log
(state, action, next-state) triples in the same 14/6 format that
`load_jumping_data` produces from hardware .mat files.

The controller is intentionally simple: altitude PD on z, attitude PD on
roll/pitch, then the closed-form mixer that inverts the wing-layout
torque equations from CLAUDE.md.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import mujoco

from .simulation import (
    body_name_to_id,
    actuator_name_to_id,
    get_body_pos_eul,
)
from .mpc import state_action_to_input


# ---------------------------------------------------------------------------
# PD controller
# ---------------------------------------------------------------------------

@dataclass
class PDGains:
    Kp_z: float = 0.05         # N / m
    Kd_z: float = 0.02         # N / (m/s)
    Kp_att: float = 2.0e-4     # N·m / rad   (~10× gravity tipping moment at 5° tilt)
    Kd_att: float = 2.0e-5     # N·m·s / rad
    # Cascaded lateral PD: position/velocity errors command desired tilt,
    # then attitude PD tracks the commanded tilt.
    Kp_xy: float = 1.0         # rad / m        (x error → commanded pitch, y error → roll)
    Kd_xy: float = 0.5         # rad / (m/s)
    tilt_max_rad: float = 0.175  # 10° — clamp commanded tilt magnitude


def get_body_vel_omega(mjdata):
    """Linear (world) and angular (body) velocity for the freejoint base."""
    vel = np.array(mjdata.qvel[0:3], dtype=np.float32)
    omega = np.array(mjdata.qvel[3:6], dtype=np.float32)
    return vel, omega


def mix_controls(F, Tx, Ty, L, fmax):
    """
    Solve for per-wing forces given desired total thrust + body torques.

    Wing layout (CLAUDE.md):
        f1=(+L,+L), f2=(+L,-L), f3=(-L,+L), f4=(-L,-L)
        Tx =  L*( f1 - f2 + f3 - f4)
        Ty =  L*(-f1 - f2 + f3 + f4)
        Tz =  0  (symmetric)

    System has one free dimension (the antisymmetric mode that doesn't
    affect F, Tx, Ty); we pin it to zero for a unique particular solution.
    """
    f1 = F / 4.0 + (Tx - Ty) / (4.0 * L)
    f2 = F / 4.0 - (Tx + Ty) / (4.0 * L)
    f3 = F / 4.0 + (Tx + Ty) / (4.0 * L)
    f4 = F / 4.0 + (Ty - Tx) / (4.0 * L)
    return np.clip([f1, f2, f3, f4], 0.0, fmax).astype(np.float32)


@dataclass
class HopGains:
    """Gains for the FSM (bang-bang) hopping controller."""
    z_switch_frac: float = 0.6
    F_pump: float = None
    apex_band: float = 0.005


@dataclass
class EnergyHopGains:
    """Gains for the energy-shaping hopping controller (smooth expert).

    Control law:
        F = K_E * (E_des_eff - E)
        E_des_eff = m·g·(z_des - K_apex · apex_err)     # apex feedback
        E         = 0.5·m·vz² + m·g·z
        apex_err  = last_apex − z_des                    # + when overshooting

    F is clipped to [F_min, F_max]. F_min > 0 keeps attitude authority during
    the near-apex/free-fall window where ΔE≈0.

    Apex feedback compensates for the un-modeled PE in the leg slider limit
    and contact spring — without it, the controller drifts high on the
    fixed-physics (slider-limit) hopper.
    """
    K_E: float = 300.0          # N per Joule
    F_min_frac: float = 0.0     # F_min = F_min_frac · 4·fmax (0 → pure ballistic)
    K_apex: float = 0.0         # apex-error gain. Kept for back-compat but
                                # the default 0 = memoryless expert. With
                                # proper contact damping (solref=-2000 -8.0
                                # in hopper.xml), the memoryless expert is
                                # already stable; apex feedback adds hidden
                                # state that hurts IL cloning.


class HoppingState:
    """Persistent state for the hopping controller: tracks last apex
    and previous vz sign so we can detect apex events (+ → −) across calls.
    Instantiate once per rollout."""

    __slots__ = ("last_apex", "prev_vz", "pump_enabled")

    def __init__(self, z_des):
        self.last_apex = z_des       # start optimistic — lets first cycle pump
        self.prev_vz = 0.0
        self.pump_enabled = True


def hopping_control(mjmodel, mjdata, z_des, mass, g, L, fmax,
                    att_gains: "PDGains", hop_gains: "HopGains",
                    state: "HoppingState",
                    x_des=0.0, y_des=0.0):
    """
    Apex-regulated hopping controller.

    Altitude finite-state machine with feedback on apex height:

      - Baseline thrust = mass·g (hover). Maintains attitude authority
        even during ballistic flight — the mixer can still produce Tx/Ty
        around a positive mean F.
      - If the last apex was below (z_des - apex_band) AND the robot is
        on an upward phase below z_switch, command F_pump (max thrust)
        to inject energy for the next hop.
      - If the last apex was above (z_des + apex_band), disable pumping
        on the next cycle — the robot coasts on stored energy and
        amplitude decays via contact damping until pumping re-enables.

    Attitude and lateral station-keeping use the cascaded PD from
    `pd_control` (shared att_gains). Yaw assumed 0.
    """
    pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
    vel, omega = get_body_vel_omega(mjdata)

    z = pos[2]
    vz = float(vel[2])

    # Detect apex: vz went from + to − (zero-crossing downward)
    if state.prev_vz > 0.0 and vz <= 0.0:
        state.last_apex = z
        state.pump_enabled = state.last_apex < (z_des - hop_gains.apex_band)
    elif state.last_apex > (z_des + hop_gains.apex_band):
        state.pump_enabled = False
    elif state.last_apex < (z_des - hop_gains.apex_band):
        state.pump_enabled = True
    state.prev_vz = vz

    z_switch = hop_gains.z_switch_frac * z_des
    F_pump = hop_gains.F_pump if hop_gains.F_pump is not None else 4.0 * fmax

    if state.pump_enabled and vz > 0.0 and z < z_switch:
        F_cmd = F_pump
    else:
        # Ballistic during flight: zero net thrust + gravity = natural arc.
        # Attitude mixer still produces some asymmetric wing forces from
        # Tx/Ty when F_cmd=0 (negative f_i are clipped); those happen to
        # act as a small positive total thrust, which is harmless here.
        F_cmd = 0.0

    pitch_cmd = -att_gains.Kp_xy * (pos[0] - x_des) - att_gains.Kd_xy * vel[0]
    roll_cmd = att_gains.Kp_xy * (pos[1] - y_des) + att_gains.Kd_xy * vel[1]
    pitch_cmd = float(np.clip(pitch_cmd, -att_gains.tilt_max_rad, att_gains.tilt_max_rad))
    roll_cmd = float(np.clip(roll_cmd, -att_gains.tilt_max_rad, att_gains.tilt_max_rad))

    Tx = -att_gains.Kp_att * (eul[0] - roll_cmd) - att_gains.Kd_att * omega[0]
    Ty = -att_gains.Kp_att * (eul[1] - pitch_cmd) - att_gains.Kd_att * omega[1]

    return mix_controls(F_cmd, Tx, Ty, L, fmax)


def energy_hopping_control(mjmodel, mjdata, z_des, mass, g, L, fmax,
                           att_gains: "PDGains",
                           energy_gains: "EnergyHopGains",
                           state: "HoppingState" = None,
                           x_des=0.0, y_des=0.0):
    """
    Energy-shaping hopping controller (smooth alternative to the FSM version).

    Total mechanical energy E = 0.5·m·vz² + m·g·z; target E_des = m·g·z_des_eff
    (where z_des_eff lowers the target when the previous apex overshot).
    Thrust = K_E · (E_des_eff − E), clipped to [F_min, 4·fmax].

    The memory-less version (state=None) is stable only on the original
    physics where the slider telescoped and dissipated energy. On the
    fixed-limits physics the apex drifts high; passing a HoppingState
    enables apex-error feedback:

        z_des_eff = z_des − K_apex · (last_apex − z_des)

    Attitude and lateral station-keeping reuse the cascaded PD from
    `pd_control`.
    """
    pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
    vel, omega = get_body_vel_omega(mjdata)

    z = float(pos[2])
    vz = float(vel[2])

    if state is not None:
        # Detect apex (vz crosses + → −) and update last_apex.
        if state.prev_vz > 0.0 and vz <= 0.0:
            state.last_apex = z
        state.prev_vz = vz
        apex_err = state.last_apex - z_des
        z_des_eff = max(0.005, z_des - energy_gains.K_apex * apex_err)
    else:
        z_des_eff = z_des

    E = 0.5 * mass * vz * vz + mass * g * z
    E_des = mass * g * z_des_eff
    F_cmd = energy_gains.K_E * (E_des - E)

    F_min = energy_gains.F_min_frac * 4.0 * fmax
    F_cmd = float(np.clip(F_cmd, F_min, 4.0 * fmax))

    pitch_cmd = -att_gains.Kp_xy * (pos[0] - x_des) - att_gains.Kd_xy * vel[0]
    roll_cmd = att_gains.Kp_xy * (pos[1] - y_des) + att_gains.Kd_xy * vel[1]
    pitch_cmd = float(np.clip(pitch_cmd, -att_gains.tilt_max_rad, att_gains.tilt_max_rad))
    roll_cmd = float(np.clip(roll_cmd, -att_gains.tilt_max_rad, att_gains.tilt_max_rad))

    Tx = -att_gains.Kp_att * (eul[0] - roll_cmd) - att_gains.Kd_att * omega[0]
    Ty = -att_gains.Kp_att * (eul[1] - pitch_cmd) - att_gains.Kd_att * omega[1]

    return mix_controls(F_cmd, Tx, Ty, L, fmax)


def pd_control(mjmodel, mjdata, z_des, mass, g, L, fmax, gains: PDGains,
               x_des=0.0, y_des=0.0):
    """
    Cascaded PD: lateral position/velocity errors command a desired tilt;
    the attitude PD then tracks that tilt. Altitude PD sets total thrust.

    With small-angle body frame, thrust direction in world:
        fx_world = F * sin(pitch),   fy_world = -F * sin(roll)
    so a +x error (body east of target) wants pitch < 0 to pull it back,
    and a +y error wants roll > 0.
    """
    pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
    vel, omega = get_body_vel_omega(mjdata)

    F_hover = mass * g
    F_cmd = F_hover + gains.Kp_z * (z_des - pos[2]) - gains.Kd_z * vel[2]
    F_cmd = float(np.clip(F_cmd, 0.0, 4.0 * fmax))

    pitch_cmd = -gains.Kp_xy * (pos[0] - x_des) - gains.Kd_xy * vel[0]
    roll_cmd = gains.Kp_xy * (pos[1] - y_des) + gains.Kd_xy * vel[1]
    pitch_cmd = float(np.clip(pitch_cmd, -gains.tilt_max_rad, gains.tilt_max_rad))
    roll_cmd = float(np.clip(roll_cmd, -gains.tilt_max_rad, gains.tilt_max_rad))

    Tx = -gains.Kp_att * (eul[0] - roll_cmd) - gains.Kd_att * omega[0]
    Ty = -gains.Kp_att * (eul[1] - pitch_cmd) - gains.Kd_att * omega[1]

    return mix_controls(F_cmd, Tx, Ty, L, fmax)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    X: np.ndarray          # [N, 14]
    Y: np.ndarray          # [N, 6]
    pos: np.ndarray        # [N+1, 3]
    eul: np.ndarray        # [N+1, 3]
    healthy: bool          # False if attitude blew up or left the arena


def randomize_initial_state(mjmodel, mjdata, rng, z_range=(0.03, 0.10),
                            tilt_deg=5.0, xy_range=0.05):
    """Reset state and place the hopper at a randomized pose."""
    mujoco.mj_resetData(mjmodel, mjdata)

    x0 = rng.uniform(-xy_range, xy_range)
    y0 = rng.uniform(-xy_range, xy_range)
    z0 = rng.uniform(*z_range)
    roll = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
    pitch = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
    # Yaw fixed at 0: the PD's cascaded lateral control treats world x/y
    # as if they match body pitch/roll axes, which is only correct at yaw=0.
    yaw = 0.0

    # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    mjdata.qpos[:7] = [x0, y0, z0, qw, qx, qy, qz]
    mjdata.qvel[:] = 0.0
    mujoco.mj_forward(mjmodel, mjdata)


def run_rollout(
    mjmodel,
    mjdata,
    z_des_fn: Callable[[float], float],
    duration: float = 5.0,
    fps_ctrl: float = 100.0,
    action_noise: float = 0.05,
    gains: Optional[PDGains] = None,
    fmax: float = 0.003,
    L: float = 0.015,
    rng: Optional[np.random.Generator] = None,
) -> RolloutResult:
    """
    Run a single MuJoCo rollout under PD control and return (X, Y).

    Args:
        mjmodel, mjdata: already-reset MuJoCo model & data.
        z_des_fn: callable(t) -> desired z for controller at time t (seconds).
        duration: rollout length in seconds.
        fps_ctrl: control rate (Hz). Matches hardware data (100 Hz).
        action_noise: std of additive wing-force noise as a fraction of fmax.
        gains: PDGains. Default gains are used if None.
        fmax: per-wing thrust cap (N).
        L: wing moment arm (m).
        rng: numpy Generator for action noise; created if None.

    Returns:
        RolloutResult with X [N,14], Y [N,6], full pos/eul trace, and a
        healthy flag (False if attitude flipped or went out of arena).
    """
    if gains is None:
        gains = PDGains()
    if rng is None:
        rng = np.random.default_rng()

    dt_ctrl = 1.0 / fps_ctrl
    n_steps = int(duration * fps_ctrl)
    sub_steps = max(1, int(round(dt_ctrl / mjmodel.opt.timestep)))
    mass = float(np.sum(mjmodel.body_mass))
    g = float(-mjmodel.opt.gravity[2])

    actuator_ids = [actuator_name_to_id(mjmodel, n) for n in ("f1", "f2", "f3", "f4")]

    pos_log = np.zeros((n_steps + 1, 3), dtype=np.float32)
    eul_log = np.zeros((n_steps + 1, 3), dtype=np.float32)
    u_log = np.zeros((n_steps, 4), dtype=np.float32)

    p0, e0 = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
    pos_log[0] = p0
    eul_log[0] = e0

    healthy = True
    for i in range(n_steps):
        t = i * dt_ctrl
        u = pd_control(mjmodel, mjdata, z_des_fn(t), mass, g, L, fmax, gains)
        if action_noise > 0:
            u = np.clip(
                u + rng.normal(0.0, action_noise * fmax, size=4),
                0.0, fmax,
            ).astype(np.float32)
        u_log[i] = u

        for j, aid in enumerate(actuator_ids):
            mjdata.ctrl[aid] = float(u[j])
        for _ in range(sub_steps):
            mujoco.mj_step(mjmodel, mjdata)

        p, e = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
        pos_log[i + 1] = p
        eul_log[i + 1] = e

        # Health check: robot didn't flip or leave the arena
        if (abs(e[0]) > np.deg2rad(60.0)
                or abs(e[1]) > np.deg2rad(60.0)
                or abs(p[0]) > 0.5
                or abs(p[1]) > 0.5):
            healthy = False
            # Truncate logs here and return partial data? For simplicity,
            # we keep the logs (some samples before the blow-up are still
            # usable) but flag unhealthy so the caller can drop the whole
            # rollout if it wants.
            break

    # If we broke early, keep only the steps we actually ran
    valid_n = i + 1 if not healthy else n_steps
    pos_log = pos_log[: valid_n + 1]
    eul_log = eul_log[: valid_n + 1]
    u_log = u_log[:valid_n]

    X = np.stack([
        state_action_to_input(pos_log[k], eul_log[k], u_log[k])
        for k in range(valid_n)
    ]).astype(np.float32)
    Y = np.concatenate(
        [pos_log[1:] - pos_log[:-1], eul_log[1:] - eul_log[:-1]],
        axis=1,
    ).astype(np.float32)

    return RolloutResult(X=X, Y=Y, pos=pos_log, eul=eul_log, healthy=healthy)


# ---------------------------------------------------------------------------
# Setpoint profiles
# ---------------------------------------------------------------------------

def make_hover_profile(z_target, jitter_std=0.005, period=0.5, rng=None):
    """Mostly constant z_des with low-amplitude drift around z_target."""
    if rng is None:
        rng = np.random.default_rng()
    phase = rng.uniform(0, 2 * np.pi)

    def z_des(t):
        return float(z_target + jitter_std * np.sin(2 * np.pi * t / period + phase))
    return z_des


def make_step_profile(z_low=0.02, z_high=0.12, dwell=0.5, rng=None):
    """Alternate between two target heights every `dwell` seconds."""
    if rng is None:
        rng = np.random.default_rng()
    start_high = bool(rng.integers(0, 2))

    def z_des(t):
        n = int(t // dwell)
        hi = (n % 2 == 0) == start_high
        return z_high if hi else z_low
    return z_des
