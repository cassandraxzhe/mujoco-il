"""
Gymnasium environment for RL training of the hopper.

Observation (11-dim, same layout as the IL policy):
    [x − x_des, y − y_des, z,
     vx, vy, vz,
     roll, pitch, ω_roll, ω_pitch,
     z_des]

Action (3-dim, normalized to [-1, 1]):
    [F_norm, Tx_norm, Ty_norm]
where
    F  = (F_norm + 1) / 2 · 4·fmax          ∈ [0, 4·fmax]
    Tx =  Tx_norm · 2·L·fmax                ∈ ± 2·L·fmax
    Ty =  Ty_norm · 2·L·fmax                ∈ ± 2·L·fmax
The analytical mixer (same one the IL's ftxty policy uses) turns
(F, Tx, Ty) into per-wing forces, which are then applied to MuJoCo.
This mirrors the IL pipeline exactly, so RL learns the same control
parameterisation and the resulting comparison is apples-to-apples.

Reward per step:
    r =   + 1.0                              # alive bonus
        − w_z   · (z − z_des)²               # altitude tracking
        − w_x   · (x − x_des)²               # lateral tracking
        − w_att · (roll² + pitch²)           # attitude stability
        − w_u   · ||(f1, f2, f3, f4) / fmax||² / 4   # control effort
        + w_fwd · max(0, x − x_prev)         # forward-progress bonus
On termination (|roll| or |pitch| > 60° or |x|, |y| > 0.5 m) the
agent receives a terminal penalty and the episode ends.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
import mujoco


@dataclass
class HopperEnvConfig:
    xml_path: str = "assets/hopper.xml"
    fps_ctrl: float = 100.0              # Hz
    max_episode_seconds: float = 5.0
    fmax: float = 0.003                  # per-wing force cap (N)
    L_arm: float = 0.015                 # wing moment arm (m)

    # Task
    z_des: float = 0.08                  # altitude target (m)
    vx_des: float = 0.0                  # forward velocity target (m/s)
    z_des_range: tuple = (0.06, 0.10)    # randomised per-episode if not None
    vx_des_range: tuple = (0.01, 0.05)   # strictly positive in training
    randomize_task: bool = True

    # Reward weights.
    #
    # A pure z-tracking penalty is counter-productive for a hopping task:
    # the expert's z oscillates between 0 (ground contact) and z_des (apex)
    # so perfect tracking is impossible. Instead of penalising z_err
    # quadratically, we only penalise being OUTSIDE a [z_floor, z_ceil]
    # corridor — "don't crash, don't rocket" — and reward the bounded
    # alive-plus-forward objective inside that corridor. Under this scheme
    # the expert's hopping cycle scores near 0 cost and a sky-rocketing
    # policy takes a large hit the moment z > z_ceil.
    w_alive: float = 1.0
    w_x: float = 300.0       # at x_err=5 cm costs 0.75/step — dominates alive bonus
    w_y: float = 300.0       # symmetric y-tracking so the policy can't drift laterally
    w_att: float = 3.0
    w_u: float = 0.02
    w_fwd: float = 10.0
    w_terminal_tip: float = 100.0
    x_err_terminal: float = 0.15  # |x - x_des| beyond this ends the episode
    y_err_terminal: float = 0.10  # |y| beyond this ends the episode
    # Altitude corridor
    z_floor_hard: float = 0.005     # below → out-of-corridor penalty scales in
    z_ceil_scale: float = 1.5       # ceiling = z_ceil_scale · z_des
    w_z_oob: float = 500.0          # penalty weight for z outside corridor

    # Termination thresholds
    tip_angle_rad: float = float(np.deg2rad(60.0))
    arena_half: float = 0.5              # |x|, |y| must stay within (m)

    # Initial state randomisation (mild — preserves expert-style trajectories)
    tilt_deg_init: float = 0.0
    xy_range_init: float = 0.0
    z_init_lo: float = 0.035
    z_init_hi: float = 0.035

    # Domain randomisation (applied at reset for sim-to-real robustness).
    #   dr_damping_range: multiplicative scaling of the leg-contact damping
    #     component of solref (solref[1], which is negative → magnitude).
    #   dr_wing_gain_range: multiplicative scaling of each wing actuator's
    #     gear value (equivalent to scaling thrust per unit ctrl).
    # Both are uniform samples in [lo, hi]; set both bounds to 1.0 to disable.
    dr_damping_range: tuple = (1.0, 1.0)
    dr_wing_gain_range: tuple = (1.0, 1.0)


def _mix_wings(F, Tx, Ty, L, fmax):
    """Analytical mixer identical to hopper.sim_data_collection.mix_controls."""
    inv = 1.0 / (4.0 * L)
    f1 = F / 4.0 + (Tx - Ty) * inv
    f2 = F / 4.0 - (Tx + Ty) * inv
    f3 = F / 4.0 + (Tx + Ty) * inv
    f4 = F / 4.0 + (Ty - Tx) * inv
    return np.clip([f1, f2, f3, f4], 0.0, fmax).astype(np.float32)


class HopperEnv(gym.Env):
    """MuJoCo hopper wrapped as a Gymnasium environment for PPO.

    One physics stepping cycle per env.step(): apply the mixer-derived wing
    forces, advance the sim by 1 / fps_ctrl seconds (possibly many mujoco
    substeps), then return the next observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[HopperEnvConfig] = None):
        super().__init__()
        self.cfg = cfg or HopperEnvConfig()

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self._mass = float(np.sum(self.model.body_mass))
        self._g = float(-self.model.opt.gravity[2])
        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hopper",
        )
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ("f1", "f2", "f3", "f4")
        ]

        # Cache nominal DR targets so we can re-seed from nominal at each reset.
        self._leg_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "leg",
        )
        if self._leg_geom_id >= 0:
            self._leg_solref_nominal = self.model.geom_solref[self._leg_geom_id].copy()
        else:
            self._leg_solref_nominal = None
        self._actuator_gear_nominal = self.model.actuator_gear[self._actuator_ids].copy()

        # Control cadence
        self._dt_ctrl = 1.0 / self.cfg.fps_ctrl
        self._substeps = max(1, int(round(self._dt_ctrl / self.model.opt.timestep)))
        self._max_steps = int(self.cfg.max_episode_seconds * self.cfg.fps_ctrl)

        # Action ranges (physical)
        self._F_max = 4.0 * self.cfg.fmax
        self._T_max = 2.0 * self.cfg.L_arm * self.cfg.fmax

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )
        # Observation: generous bounds (not binding — we clip nowhere).
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(11,), dtype=np.float32,
        )

        self._rng = np.random.default_rng(0)
        self._step_count = 0
        self._z_des = self.cfg.z_des
        self._vx_des = self.cfg.vx_des
        self._prev_x = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Domain randomisation of physical parameters — applied ONLY if the
        # range bounds aren't both 1.0 (i.e., DR explicitly enabled).
        if self._leg_solref_nominal is not None:
            lo, hi = self.cfg.dr_damping_range
            if lo != 1.0 or hi != 1.0:
                scale = float(self._rng.uniform(lo, hi))
                # Leg contact solref = [stiffness, damping] in negative form
                # (both values stored as negatives). Scale the damping term.
                self.model.geom_solref[self._leg_geom_id] = (
                    self._leg_solref_nominal * np.array([1.0, scale], dtype=np.float64)
                )
        lo, hi = self.cfg.dr_wing_gain_range
        if lo != 1.0 or hi != 1.0:
            scale = float(self._rng.uniform(lo, hi))
            self.model.actuator_gear[self._actuator_ids] = (
                self._actuator_gear_nominal * scale
            )

        # Randomise initial pose (mild, matches collect_il_demos defaults).
        x0 = self._rng.uniform(-self.cfg.xy_range_init, self.cfg.xy_range_init)
        y0 = self._rng.uniform(-self.cfg.xy_range_init, self.cfg.xy_range_init)
        z0 = self._rng.uniform(self.cfg.z_init_lo, self.cfg.z_init_hi)
        roll = np.deg2rad(self._rng.uniform(-self.cfg.tilt_deg_init, self.cfg.tilt_deg_init))
        pitch = np.deg2rad(self._rng.uniform(-self.cfg.tilt_deg_init, self.cfg.tilt_deg_init))
        yaw = 0.0  # kept 0: the cascaded-PD conventions assume yaw=0.

        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        self.data.qpos[:7] = [x0, y0, z0, qw, qx, qy, qz]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Randomise task targets (per-episode). Keep vx_des strictly
        # positive during training — the alive bonus + no-motion case lets
        # a passive "lazy" policy score ~500 and hide the real task.
        if self.cfg.randomize_task:
            self._z_des = float(self._rng.uniform(*self.cfg.z_des_range))
            self._vx_des = float(self._rng.uniform(*self.cfg.vx_des_range))
        else:
            self._z_des = self.cfg.z_des
            self._vx_des = self.cfg.vx_des

        self._step_count = 0
        self._prev_x = float(self.data.xpos[self._body_id][0])
        return self._obs(), {}

    def step(self, action):
        # Unpack and denormalise action.
        a = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        F = (a[0] + 1.0) * 0.5 * self._F_max
        Tx = a[1] * self._T_max
        Ty = a[2] * self._T_max
        wings = _mix_wings(F, Tx, Ty, self.cfg.L_arm, self.cfg.fmax)

        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = float(wings[i])
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._obs()
        reward, terminated, truncated = self._reward_and_done(wings)
        info = {"z_des": self._z_des, "vx_des": self._vx_des,
                "x": float(self.data.xpos[self._body_id][0]),
                "z": float(self.data.xpos[self._body_id][2])}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_targets(self):
        """x_des advances linearly at vx_des; y_des fixed at 0."""
        t = self._step_count * self._dt_ctrl
        return self._vx_des * t, 0.0

    def _obs(self):
        pos = self.data.xpos[self._body_id]
        # Euler ZYX (roll, pitch, yaw) — same derivation as simulation.get_body_pos_eul
        R = np.asarray(self.data.xmat).reshape(-1, 3, 3)[self._body_id]
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        # qvel layout for freejoint: [vx, vy, vz (world), wx, wy, wz (body)]
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        vz = float(self.data.qvel[2])
        wr = float(self.data.qvel[3])
        wp = float(self.data.qvel[4])

        x_des, y_des = self._current_targets()

        return np.array([
            pos[0] - x_des, pos[1] - y_des, pos[2],
            vx, vy, vz,
            roll, pitch, wr, wp,
            self._z_des,
        ], dtype=np.float32)

    def _reward_and_done(self, wings):
        pos = self.data.xpos[self._body_id]
        z = float(pos[2]); x = float(pos[0]); y = float(pos[1])

        R = np.asarray(self.data.xmat).reshape(-1, 3, 3)[self._body_id]
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])

        x_des, y_des = self._current_targets()

        # Control-effort term (normalise by fmax so this weight scales cleanly)
        u_cost = float(np.sum((wings / self.cfg.fmax) ** 2)) / 4.0

        # Forward-progress bonus: how much x advanced this step.
        # Capped at 0 so the agent doesn't get rewarded for drifting backward;
        # no upper cap — running past x_des is still forward progress.
        fwd = max(0.0, x - self._prev_x)
        self._prev_x = x

        # Corridor penalty: z outside [z_floor_hard, z_ceil] gets a quadratic
        # hit. Alive bonus is gated by a smooth sigmoid on (z − 2 cm) so the
        # policy can't "farm" alive reward by lying on the ground.
        z_ceil = self.cfg.z_ceil_scale * self._z_des
        z_oob = max(0.0, self.cfg.z_floor_hard - z) + max(0.0, z - z_ceil)
        alive_gate = 1.0 / (1.0 + np.exp(-(z - 0.02) / 0.005))  # ≈0 at z=0, ≈1 at z>3cm
        r = (self.cfg.w_alive * alive_gate
             - self.cfg.w_z_oob * z_oob ** 2
             - self.cfg.w_x     * (x - x_des) ** 2
             - self.cfg.w_y     * (y - y_des) ** 2
             - self.cfg.w_att   * (roll ** 2 + pitch ** 2)
             - self.cfg.w_u     * u_cost
             + self.cfg.w_fwd   * fwd)

        terminated = False
        truncated = False
        if (abs(roll) > self.cfg.tip_angle_rad
                or abs(pitch) > self.cfg.tip_angle_rad
                or abs(x) > self.cfg.arena_half
                or abs(y) > self.cfg.arena_half):
            r -= self.cfg.w_terminal_tip
            terminated = True
        elif (abs(x - x_des) > self.cfg.x_err_terminal
                or abs(y - y_des) > self.cfg.y_err_terminal):
            # Drift too far from the xy-setpoint (passive policies get killed here).
            r -= self.cfg.w_terminal_tip
            terminated = True
        elif self._step_count >= self._max_steps:
            truncated = True

        return float(r), terminated, truncated

    def close(self):
        pass
