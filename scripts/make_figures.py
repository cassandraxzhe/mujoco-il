"""
Generate thesis-ready figures for the IL-vs-PPO-vs-expert comparison.

Produces five figures under experiments/figures/:

  (a) bar_comparison.pdf  — max x and episode length per controller × scenario
  (b) com_trajectory.pdf  — x-vs-z traces of expert / IL / PPO on the 3-step flight,
                            with stair geometry drawn as background
  (c) learning_curves.pdf — PPO reward + episode length over timesteps for the
                            three training configurations
  (d) hop_cycles.pdf      — z-vs-time showing the hopping gait of each controller
                            on flat forward motion (vx=3 cm/s)
  (e) stair_schematic.pdf — labelled side view of the 3-step flight geometry

All figures use a consistent color palette:
    expert    = grey
    IL        = blue
    PPO (any) = orange variants
"""

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hopper.rl_env import HopperEnv, HopperEnvConfig, _mix_wings
from hopper.il_policy import ILPolicyFTxTy, extract_il_state, IL_STATE_DIM
from hopper.sim_data_collection import (
    PDGains, EnergyHopGains, energy_hopping_control,
)
from hopper.simulation import body_name_to_id, actuator_name_to_id


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

COLORS = {
    "expert":         "#555555",
    "il_fwd_dag2":    "#1f77b4",
    "ppo_v2":         "#ff7f0e",
    "ppo_stair_v1":   "#d62728",
    "ppo_mix_v1":     "#2ca02c",
}
LABELS = {
    "expert":         "Expert (energy)",
    "il_fwd_dag2":    "IL (fwd_dag2)",
    "ppo_v2":         "PPO flat-only",
    "ppo_stair_v1":   "PPO stair-only",
    "ppo_mix_v1":     "PPO mix (flat+stair)",
}
CTRL_ORDER = ["expert", "il_fwd_dag2", "ppo_v2", "ppo_stair_v1", "ppo_mix_v1"]

OUT_DIR = "experiments/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Rollout helpers (reuse structure from scripts/compare_controllers.py)
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    xml: str
    z_des: float
    vx_des: float
    x0: float
    sim_time: float = 5.0


def _log_wings(u, storage):
    storage.append(np.array(u, dtype=np.float32).copy())


def rollout_expert(scn, return_wings=False):
    m = mujoco.MjModel.from_xml_path(scn.xml); d = mujoco.MjData(m)
    d.qpos[:3] = [scn.x0, 0.0, 0.035]; d.qpos[3:7] = [1.0, 0, 0, 0]
    mujoco.mj_forward(m, d)
    mass = float(np.sum(m.body_mass)); g = float(-m.opt.gravity[2])
    aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
    spf = max(1, int(round(0.01 / m.opt.timestep)))
    bid = body_name_to_id(m, "hopper")
    att = PDGains(); eg = EnergyHopGains()
    xs, ys, zs, ws = [], [], [], []
    for i in range(int(scn.sim_time * 100)):
        t = i * 0.01
        u = energy_hopping_control(m, d, scn.z_des, mass, g, 0.015, 0.003,
                                   att, eg, x_des=scn.vx_des*t, y_des=0)
        ws.append(np.array(u, dtype=np.float32))
        for j, aid in enumerate(aids): d.ctrl[aid] = float(u[j])
        for _ in range(spf): mujoco.mj_step(m, d)
        p = d.xpos[bid]; xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        R = np.asarray(d.xmat).reshape(-1, 3, 3)[bid]
        pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
        roll = np.arctan2(R[2,1], R[2,2])
        if abs(roll) > np.deg2rad(60) or abs(pitch) > np.deg2rad(60):
            break
    t_arr = np.arange(len(xs)) * 0.01
    xs_a = np.array(xs); ys_a = np.array(ys); zs_a = np.array(zs); ws_a = np.array(ws)
    return (t_arr, xs_a, ys_a, zs_a, ws_a) if return_wings else (t_arr, xs_a, ys_a, zs_a)


def rollout_il(weights, norm_npz, scn, return_wings=False):
    norm = np.load(norm_npz)
    X_mean = norm["X_mean"].astype(np.float32); X_std = norm["X_std"].astype(np.float32)
    policy = ILPolicyFTxTy(input_dim=IL_STATE_DIM, hidden_dim=64)
    policy.load(weights); policy.eval()
    m = mujoco.MjModel.from_xml_path(scn.xml); d = mujoco.MjData(m)
    d.qpos[:3] = [scn.x0, 0.0, 0.035]; d.qpos[3:7] = [1.0, 0, 0, 0]
    mujoco.mj_forward(m, d)
    aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
    spf = max(1, int(round(0.01 / m.opt.timestep)))
    bid = body_name_to_id(m, "hopper")
    xs, ys, zs, ws = [], [], [], []
    for i in range(int(scn.sim_time * 100)):
        t = i * 0.01
        s = extract_il_state(m, d, scn.z_des, x_des=scn.vx_des*t, y_des=0)
        s_n = (s - X_mean) / X_std
        with torch.no_grad():
            u = policy.wing_forces(torch.tensor(s_n).unsqueeze(0)).numpy()[0]
        ws.append(np.array(u, dtype=np.float32))
        for j, aid in enumerate(aids): d.ctrl[aid] = float(u[j])
        for _ in range(spf): mujoco.mj_step(m, d)
        p = d.xpos[bid]; xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        R = np.asarray(d.xmat).reshape(-1, 3, 3)[bid]
        pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
        roll = np.arctan2(R[2,1], R[2,2])
        if abs(roll) > np.deg2rad(60) or abs(pitch) > np.deg2rad(60):
            break
    t_arr = np.arange(len(xs)) * 0.01
    xs_a = np.array(xs); ys_a = np.array(ys); zs_a = np.array(zs); ws_a = np.array(ws)
    return (t_arr, xs_a, ys_a, zs_a, ws_a) if return_wings else (t_arr, xs_a, ys_a, zs_a)


def rollout_ppo(run_name, scn, return_wings=False):
    rundir = os.path.join("experiments/ppo", run_name)
    cfg = HopperEnvConfig(
        xml_path=scn.xml, max_episode_seconds=scn.sim_time,
        z_des=scn.z_des, vx_des=scn.vx_des, randomize_task=False,
        z_init_lo=0.035, z_init_hi=0.035, xy_range_init=0.0, tilt_deg_init=0.0,
    )

    def _factory():
        e = HopperEnv(cfg); e.reset()
        e.data.qpos[0] = scn.x0; e.data.qpos[1] = 0.0; e.data.qpos[2] = 0.035
        e.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(e.model, e.data)
        return e

    venv = DummyVecEnv([_factory])
    venv = VecNormalize.load(os.path.join(rundir, "vec_normalize.pkl"), venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(os.path.join(rundir, "model.zip"), env=venv, device="cpu")
    env = venv.envs[0].unwrapped
    bid = body_name_to_id(env.model, "hopper")
    obs = venv.reset()
    env.data.qpos[0] = scn.x0; env.data.qpos[1] = 0.0; env.data.qpos[2] = 0.035
    env.data.qpos[3:7] = [1, 0, 0, 0]; env.data.qvel[:] = 0
    env._step_count = 0; env._prev_x = scn.x0
    env._z_des = scn.z_des; env._vx_des = scn.vx_des
    mujoco.mj_forward(env.model, env.data)
    xs, ys, zs, ws = [], [], [], []
    for i in range(int(scn.sim_time * 100)):
        xs.append(env.data.xpos[bid][0]); ys.append(env.data.xpos[bid][1])
        zs.append(env.data.xpos[bid][2])
        action, _ = model.predict(obs, deterministic=True)
        # Recover wing forces from the action via the mixer (same math as env).
        a = np.asarray(action[0]).clip(-1, 1)
        F = (a[0] + 1.0) * 0.5 * env._F_max
        Tx = a[1] * env._T_max
        Ty = a[2] * env._T_max
        ws.append(_mix_wings(F, Tx, Ty, env.cfg.L_arm, env.cfg.fmax))
        obs, reward, done, info = venv.step(action)
        if done[0]:
            if "x" in info[0]:
                xs.append(info[0]["x"]); zs.append(info[0]["z"]); ys.append(ys[-1])
            break
    t_arr = np.arange(len(xs)) * 0.01
    xs_a = np.array(xs); ys_a = np.array(ys); zs_a = np.array(zs); ws_a = np.array(ws)
    return (t_arr, xs_a, ys_a, zs_a, ws_a) if return_wings else (t_arr, xs_a, ys_a, zs_a)


# ---------------------------------------------------------------------------
# (a) Head-to-head bar chart
# ---------------------------------------------------------------------------

def fig_a_bar_comparison():
    scenarios = [
        Scenario("flat hover",       "assets/hopper.xml", 0.08, 0.00,  0.00),
        Scenario("flat fwd",         "assets/hopper.xml", 0.08, 0.03, -0.03),
        Scenario("single step",      "assets/hopper_stair_h8_d20.xml", 0.08, 0.03, -0.05),
        Scenario("3-step flight",    "assets/hopper_stair_flight_3x8mm.xml", 0.10, 0.03, -0.05, 7.0),
    ]
    # Pre-collected numbers from scripts/compare_controllers.py
    # (max x in cm, steps survived).
    data = {
        "flat hover":    {"expert": (0.0, 500),   "il_fwd_dag2": (2.0, 200),  "ppo_v2": (0.9, 500),
                          "ppo_stair_v1": (6.9, 500), "ppo_mix_v1": (3.4, 500)},
        "flat fwd":      {"expert": (196.8, 456), "il_fwd_dag2": (5.4, 180),  "ppo_v2": (14.9, 500),
                          "ppo_stair_v1": (16.7, 500), "ppo_mix_v1": (15.0, 500)},
        "single step":   {"expert": (99.2, 298),  "il_fwd_dag2": (77.7, 222), "ppo_v2": (10.9, 265),
                          "ppo_stair_v1": (9.7, 159),  "ppo_mix_v1": (11.4, 447)},
        "3-step flight": {"expert": (20.7, 186),  "il_fwd_dag2": (22.0, 113), "ppo_v2": (10.7, 268),
                          "ppo_stair_v1": (13.0, 654), "ppo_mix_v1": (11.9, 600)},
    }

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    n_scn = len(scenarios)
    n_ctrl = len(CTRL_ORDER)
    width = 0.16
    x = np.arange(n_scn)

    for i, ctrl in enumerate(CTRL_ORDER):
        offsets = x + (i - (n_ctrl - 1) / 2) * width
        max_xs = [data[scn.name][ctrl][0] for scn in scenarios]
        steps = [data[scn.name][ctrl][1] for scn in scenarios]
        axs[0].bar(offsets, max_xs, width, label=LABELS[ctrl], color=COLORS[ctrl])
        axs[1].bar(offsets, steps, width, label=LABELS[ctrl], color=COLORS[ctrl])

    axs[0].set_ylabel("Max forward distance (cm)")
    axs[0].set_title("(a) Forward reach — how far the controller gets")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([s.name for s in scenarios])
    axs[0].set_yscale("symlog")
    # Crossed-flight annotation
    flight_idx = 3
    for i, ctrl in enumerate(CTRL_ORDER):
        offset = x[flight_idx] + (i - (n_ctrl - 1) / 2) * width
        mx = data["3-step flight"][ctrl][0]
        if mx > 11.0:  # step 3 ends at 11 cm
            axs[0].text(offset, mx * 1.12, "✓", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="black")
    axs[0].axhline(11, linestyle=":", color="grey", linewidth=0.8,
                   label="flight end (x=11 cm)")

    axs[1].set_ylabel("Episode length (time-steps)")
    axs[1].set_title("(b) Stability — steps survived (max 500 for 5 s / 700 for 7 s)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([s.name for s in scenarios])

    axs[1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    fig.suptitle("Head-to-head controller comparison across four scenarios",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "bar_comparison.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Stair geometry (shared by b and e)
# ---------------------------------------------------------------------------

def draw_stair_flight(ax, heights=(0.008, 0.008, 0.008),
                      depths=(0.020, 0.020, 0.020), x_offset=0.05,
                      face="#d0d0d0", edge="#808080"):
    """Draw a 3-step cumulative flight as filled rectangles."""
    cum_x = x_offset
    for h, d in zip(heights, depths):
        # This step's top is cumulative; its box starts at the floor and ends
        # at the cumulative top.
        top_z = sum(heights[: heights.index(h) + 1]) if len(set(heights)) > 1 else None
        # Using cumulative-top index based on position, safer:
        pass
    # Simpler: re-derive cumulative tops.
    cum_x = x_offset
    cum_top = 0.0
    for h, d in zip(heights, depths):
        cum_top += h
        rect = Rectangle((cum_x * 100, 0), d * 100, cum_top * 100,
                         facecolor=face, edgecolor=edge, linewidth=0.8)
        ax.add_patch(rect)
        cum_x += d


# ---------------------------------------------------------------------------
# (b) COM trajectory overlays
# ---------------------------------------------------------------------------

def fig_b_com_trajectory():
    scn = Scenario("3-step flight",
                   "assets/hopper_stair_flight_3x8mm.xml",
                   z_des=0.10, vx_des=0.03, x0=-0.05, sim_time=7.0)

    rollouts = {}
    rollouts["expert"]       = rollout_expert(scn)
    rollouts["il_fwd_dag2"]  = rollout_il(
        "experiments/weights/il_energy_ftxty_fwd_dag2.pt",
        "experiments/weights/il_energy_ftxty_fwd_dag2_norm.npz", scn)
    rollouts["ppo_v2"]       = rollout_ppo("ppo_v2", scn)
    rollouts["ppo_stair_v1"] = rollout_ppo("ppo_stair_v1", scn)
    rollouts["ppo_mix_v1"]   = rollout_ppo("ppo_mix_v1", scn)

    fig, axs = plt.subplots(2, 1, figsize=(10, 7),
                            gridspec_kw={"height_ratios": [1.4, 1]})
    # Top: x-z side view with stair geometry.
    draw_stair_flight(axs[0])
    for ctrl, (t, xs, ys, zs) in rollouts.items():
        axs[0].plot(xs * 100, zs * 100, lw=1.5,
                    color=COLORS[ctrl], label=LABELS[ctrl])
        axs[0].scatter(xs[-1] * 100, zs[-1] * 100, s=40, marker="X",
                       color=COLORS[ctrl], zorder=5, edgecolors="white")
    # Stair edges as vertical dashes
    for x_edge in [5, 7, 9, 11]:
        axs[0].axvline(x_edge, linestyle=":", color="#888", linewidth=0.6)
    axs[0].set_ylabel("z (cm)")
    axs[0].set_title("(a) COM side view — torso x–z during 7 s climb attempt")
    axs[0].set_ylim(-0.5, 14)
    axs[0].set_xlim(-7, 30)
    axs[0].set_aspect("equal")
    axs[0].legend(loc="upper right", fontsize=8)

    # Bottom: x over time (forward progress)
    for ctrl, (t, xs, ys, zs) in rollouts.items():
        axs[1].plot(t, xs * 100, lw=1.5, color=COLORS[ctrl], label=LABELS[ctrl])
    for y_ann in [5, 7, 9, 11]:
        axs[1].axhline(y_ann, linestyle=":", color="#888", linewidth=0.6)
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("x (cm)")
    axs[1].set_title("(b) Forward progress vs time — horizontal lines mark step edges")
    axs[1].set_xlim(0, scn.sim_time)
    axs[1].grid(alpha=0.3)

    fig.suptitle(f"3-step flight (x₀={scn.x0*100:.0f} cm, vx_des={scn.vx_des*100:.0f} cm/s, z_des={scn.z_des*100:.0f} cm)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "com_trajectory.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (c) Learning curves
# ---------------------------------------------------------------------------

def _load_tb(run_name, tag):
    path = f"experiments/ppo/{run_name}/tensorboard/PPO_1"
    ea = EventAccumulator(path); ea.Reload()
    ev = ea.Scalars(tag)
    steps = np.array([e.step for e in ev])
    vals = np.array([e.value for e in ev])
    return steps, vals


def fig_c_learning_curves():
    runs = ["ppo_v2", "ppo_stair_v1", "ppo_mix_v1"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for r in runs:
        try:
            s, v = _load_tb(r, "rollout/ep_rew_mean")
            axs[0].plot(s / 1000, v, lw=1.4, color=COLORS[r], label=LABELS[r])
            s, v = _load_tb(r, "rollout/ep_len_mean")
            axs[1].plot(s / 1000, v, lw=1.4, color=COLORS[r], label=LABELS[r])
        except Exception as e:
            print(f"  (skipping {r}: {e})")

    axs[0].set_xlabel("timesteps (×10³)")
    axs[0].set_ylabel("mean episode reward")
    axs[0].set_title("(a) Reward vs training steps")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].set_xlabel("timesteps (×10³)")
    axs[1].set_ylabel("mean episode length")
    axs[1].set_title("(b) Episode length — stability proxy")
    axs[1].grid(alpha=0.3)

    fig.suptitle("PPO training curves", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "learning_curves.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (d) Hopping z-traces
# ---------------------------------------------------------------------------

def fig_d_hop_cycles():
    scn = Scenario("flat fwd vx=3", "assets/hopper.xml",
                   z_des=0.08, vx_des=0.03, x0=-0.03, sim_time=3.0)
    rollouts = {}
    rollouts["expert"]       = rollout_expert(scn)
    rollouts["il_fwd_dag2"]  = rollout_il(
        "experiments/weights/il_energy_ftxty_fwd_dag2.pt",
        "experiments/weights/il_energy_ftxty_fwd_dag2_norm.npz", scn)
    rollouts["ppo_mix_v1"]   = rollout_ppo("ppo_mix_v1", scn)

    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
    for ctrl, (t, xs, ys, zs) in rollouts.items():
        ax.plot(t, zs * 100, lw=1.3, color=COLORS[ctrl], label=LABELS[ctrl])
    ax.axhline(scn.z_des * 100, linestyle="--", color="black", linewidth=0.7,
               label=f"z_des = {scn.z_des*100:.0f} cm")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("torso z (cm)")
    ax.set_title(f"Hopping cycles on flat forward motion "
                 f"(vx_des={scn.vx_des*100:.0f} cm/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.5, 13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hop_cycles.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (e) Stair schematic
# ---------------------------------------------------------------------------

def fig_e_stair_schematic():
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.0))
    # Floor
    ax.add_patch(Rectangle((-8, -1), 30, 1, facecolor="#efefef",
                           edgecolor="#888", linewidth=0.7))
    # Flight (3 × 8 mm × 20 mm)
    draw_stair_flight(ax, face="#c8d6e5", edge="#3a4f6b")

    # Single h/d dimension annotations placed ABOVE the top step so they
    # don't collide with step labels.
    # Depth annotation on step 3 (x ∈ [9, 11] cm), shown above at z = 3.5
    ax.annotate("", xy=(11, 3.5), xytext=(9, 3.5),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(10, 3.9, "depth d = 20 mm", ha="center", fontsize=9)
    # Height annotation: arrow from floor to step-1 top at x = 4.5 (just left of stairs)
    ax.annotate("", xy=(4.5, 0.8), xytext=(4.5, 0.0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(4.3, 0.4, "h = 8 mm", ha="right", va="center", fontsize=9)

    # Step labels above each step
    for i, (x_pos, z_top) in enumerate([(6, 0.8), (8, 1.6), (10, 2.4)]):
        ax.text(x_pos, z_top + 0.15, f"step {i+1}",
                ha="center", va="bottom", fontsize=8, color="#3a4f6b")
        ax.text(x_pos, z_top - 0.2, f"z={z_top*10:.0f} mm",
                ha="center", va="top", fontsize=7, color="#3a4f6b")

    # Hopper start
    ax.scatter([-5], [3.5], s=80, marker="o", color="#c0392b", zorder=5,
               edgecolors="white", label="hopper start (x₀=−5 cm, z=3.5 cm)")
    ax.arrow(-5, 4.2, 3, 0, head_width=0.4, head_length=0.4, fc="#c0392b",
             ec="#c0392b", linewidth=1.5)
    ax.text(-3.5, 4.9, "vx_des = 3 cm/s", fontsize=9, color="#c0392b")

    # z_des reference
    ax.axhline(10, linestyle=":", color="grey", linewidth=0.8)
    ax.text(19, 10.3, "z_des = 10 cm", fontsize=9, color="grey")

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title("3-step flight schematic (each step 8 mm × 20 mm, cumulative)",
                 fontsize=11)
    ax.set_xlim(-8, 22)
    ax.set_ylim(-1, 12)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "stair_schematic.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (f) Per-wing force time-series
# ---------------------------------------------------------------------------

def fig_f_per_wing_forces():
    scn = Scenario("flat fwd vx=3", "assets/hopper.xml",
                   z_des=0.08, vx_des=0.03, x0=-0.03, sim_time=1.5)
    policies = {
        "expert":        rollout_expert(scn, return_wings=True),
        "il_fwd_dag2":   rollout_il(
            "experiments/weights/il_energy_ftxty_fwd_dag2.pt",
            "experiments/weights/il_energy_ftxty_fwd_dag2_norm.npz",
            scn, return_wings=True),
        "ppo_mix_v1":    rollout_ppo("ppo_mix_v1", scn, return_wings=True),
    }

    fig, axs = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    wing_names = ["f1 (+L,+L)", "f2 (+L,−L)", "f3 (−L,+L)", "f4 (−L,−L)"]
    for ctrl, (t, xs, ys, zs, ws) in policies.items():
        # ws has shape (len_t, 4); t may be longer if a final xy row was tacked on.
        n = len(ws)
        for wi in range(4):
            axs[wi].plot(t[:n], ws[:, wi] * 1000, lw=1.3,
                         color=COLORS[ctrl], label=LABELS[ctrl])
    for wi in range(4):
        axs[wi].set_ylabel(f"{wing_names[wi]}\n(mN)")
        axs[wi].axhline(3.0, linestyle=":", color="grey", linewidth=0.6)
        axs[wi].axhline(0.0, linestyle="-", color="grey", linewidth=0.5)
        axs[wi].grid(alpha=0.25)
        axs[wi].set_ylim(-0.2, 3.3)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("time (s)")
    axs[0].set_title("Per-wing force commands during 1.5 s of forward hopping "
                     "(vx_des=3 cm/s, z_des=8 cm)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "per_wing_forces.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (g) Apex histogram
# ---------------------------------------------------------------------------

def _detect_apexes(t_arr, zs, min_apex=0.02):
    """Return apex z-values from zero-crossings of vz (approx. first diff)."""
    if len(zs) < 3:
        return np.array([])
    vzs = np.diff(zs) / np.diff(t_arr)
    ai = np.where((vzs[:-1] > 0) & (vzs[1:] <= 0))[0]
    apex = zs[ai]
    return apex[apex > min_apex]


def fig_g_apex_histogram():
    # Run each policy for 10 s of pure hover (vx_des=0) to get many apexes.
    scn_hover = Scenario("flat hover", "assets/hopper.xml",
                         z_des=0.08, vx_des=0.0, x0=0.0, sim_time=10.0)

    results = {
        "expert":       _detect_apexes(*rollout_expert(scn_hover)[:2:-1][::-1][:2][::-1])
        if False else None,  # no — rebuild below for clarity
    }
    # Cleaner: unpack properly.
    results = {}
    for name, loader in [
        ("expert",      lambda: rollout_expert(scn_hover)),
        ("il_fwd_dag2", lambda: rollout_il(
            "experiments/weights/il_energy_ftxty_fwd_dag2.pt",
            "experiments/weights/il_energy_ftxty_fwd_dag2_norm.npz",
            scn_hover)),
        ("ppo_mix_v1",  lambda: rollout_ppo("ppo_mix_v1", scn_hover)),
    ]:
        t, xs, ys, zs = loader()
        apex = _detect_apexes(t, zs)
        results[name] = apex

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.0))
    bins = np.linspace(2, 12, 31)
    for ctrl, apex in results.items():
        if len(apex) == 0:
            continue
        ax.hist(apex * 100, bins=bins, alpha=0.55, color=COLORS[ctrl],
                label=f"{LABELS[ctrl]}  (n={len(apex)}, μ={apex.mean()*100:.2f}, σ={apex.std()*100:.2f} cm)",
                edgecolor="white", linewidth=0.3)
    ax.axvline(8.0, color="black", linestyle="--", linewidth=1.0,
               label="z_des = 8 cm")
    ax.set_xlabel("apex height (cm)")
    ax.set_ylabel("apex count")
    ax.set_title("Apex-height distribution — 10 s of flat hover at z_des=8 cm, vx_des=0")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "apex_histogram.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (h) Multi-seed learning curves
# ---------------------------------------------------------------------------

def fig_h_multi_seed(seeds=(42, 43, 44)):
    """Plot mean ± std of rollout/ep_rew_mean across seeds for ppo_mix.

    Assumes runs named ppo_mix_seed{seed} exist under experiments/ppo/.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for tag_i, (tag, ylabel, title) in enumerate([
        ("rollout/ep_rew_mean", "mean episode reward",  "(a) reward across 3 seeds"),
        ("rollout/ep_len_mean", "mean episode length",  "(b) episode length across 3 seeds"),
    ]):
        curves = []
        common_steps = None
        for s in seeds:
            run = f"ppo_mix_seed{s}"
            try:
                steps, vals = _load_tb(run, tag)
            except Exception as e:
                print(f"  (multi-seed: skipping {run}: {e})")
                continue
            # Interpolate onto a common step grid for mean/std
            if common_steps is None:
                common_steps = steps
            if len(steps) > 1:
                v_on_common = np.interp(common_steps, steps, vals,
                                        left=np.nan, right=np.nan)
                curves.append(v_on_common)
                axs[tag_i].plot(steps / 1000, vals, lw=0.8, alpha=0.4,
                                color=COLORS["ppo_mix_v1"])
        if curves:
            arr = np.array(curves)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            axs[tag_i].plot(common_steps / 1000, mean, lw=2.0,
                            color=COLORS["ppo_mix_v1"],
                            label=f"ppo_mix mean ± std  (n={len(curves)})")
            axs[tag_i].fill_between(common_steps / 1000, mean - std, mean + std,
                                    alpha=0.22, color=COLORS["ppo_mix_v1"])
        axs[tag_i].set_xlabel("timesteps (×10³)")
        axs[tag_i].set_ylabel(ylabel)
        axs[tag_i].set_title(title)
        axs[tag_i].grid(alpha=0.3)
        axs[tag_i].legend(fontsize=8)

    fig.suptitle("PPO (mix, 128², 1 M steps) — variability across 3 seeds",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "multi_seed_curves.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (i) Stair-height sweep
# ---------------------------------------------------------------------------

def fig_i_stair_height_sweep(heights_mm=(2, 4, 6, 8, 10, 12, 15, 20, 25)):
    from hopper.environments import make_stair_xml

    results = {ctrl: [] for ctrl in CTRL_ORDER}
    for h_mm in heights_mm:
        xml_path = f"/tmp/stair_sweep_h{h_mm}.xml"
        make_stair_xml(height=h_mm / 1000, depth=0.020, width=0.10,
                       x_offset=0.05, out_path=xml_path)
        scn = Scenario(f"h{h_mm}mm", xml_path,
                       z_des=0.08, vx_des=0.03, x0=-0.05, sim_time=5.0)

        for ctrl in CTRL_ORDER:
            try:
                if ctrl == "expert":
                    _, xs, _, _ = rollout_expert(scn)
                elif ctrl.startswith("il_"):
                    base = "experiments/weights/il_energy_ftxty_fwd_dag2"
                    _, xs, _, _ = rollout_il(base + ".pt", base + "_norm.npz", scn)
                else:  # PPO
                    _, xs, _, _ = rollout_ppo(ctrl, scn)
                results[ctrl].append(xs.max() * 100)
            except Exception as e:
                print(f"  sweep: {ctrl} @ h{h_mm}mm failed: {e}")
                results[ctrl].append(np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.2))
    # Step end is always at x = 7 cm (front 5 + depth 2). Anything past 7 cm = crossed.
    ax.axhline(7.0, linestyle="--", color="black", linewidth=1.0,
               label="crossed (max x > 7 cm)")
    hx = np.array(heights_mm)
    for ctrl in CTRL_ORDER:
        ys = np.array(results[ctrl])
        ax.plot(hx, ys, "o-", lw=1.6, markersize=6, color=COLORS[ctrl],
                label=LABELS[ctrl])
    ax.set_xlabel("step height (mm)")
    ax.set_ylabel("max forward x reached (cm)")
    ax.set_title("Single-step crossing vs step height "
                 "(x₀=−5 cm, vx_des=3 cm/s, step depth=20 mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(heights_mm)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "stair_height_sweep.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (j) 3-step flight robustness sweep across step heights
# ---------------------------------------------------------------------------

def fig_j_flight_height_sweep(heights_mm=(4, 6, 8, 10, 12)):
    """For each uniform 3-step flight with step height h, record each
    controller's max x reached and whether it crossed the entire flight
    (past x = x_offset + 3·depth = 5 + 3·2 = 11 cm). Robustness claim:
    the best controllers should cross all three heights."""
    from hopper.environments import make_stair_flight_xml

    sim_time = 7.0  # flight is longer — give room
    results = {ctrl: {"max_x": [], "crossed": []} for ctrl in CTRL_ORDER}
    for h_mm in heights_mm:
        h = h_mm / 1000
        xml_path = f"/tmp/stair_flight_sweep_h{h_mm}.xml"
        make_stair_flight_xml(
            heights=[h, h, h], depths=[0.020, 0.020, 0.020],
            width=0.10, x_offset=0.05, out_path=xml_path,
        )
        # Flight ends at x_offset + 3·depth = 0.11 m regardless of height.
        flight_end_cm = 11.0

        scn = Scenario(f"flight_h{h_mm}", xml_path,
                       z_des=0.10, vx_des=0.03, x0=-0.05, sim_time=sim_time)

        for ctrl in CTRL_ORDER:
            try:
                if ctrl == "expert":
                    _, xs, _, _ = rollout_expert(scn)
                elif ctrl.startswith("il_"):
                    base = "experiments/weights/il_energy_ftxty_fwd_dag2"
                    _, xs, _, _ = rollout_il(base + ".pt", base + "_norm.npz", scn)
                else:
                    _, xs, _, _ = rollout_ppo(ctrl, scn)
                max_x_cm = float(xs.max() * 100)
                results[ctrl]["max_x"].append(max_x_cm)
                results[ctrl]["crossed"].append(max_x_cm > flight_end_cm)
            except Exception as e:
                print(f"  flight-sweep: {ctrl} @ h{h_mm}mm failed: {e}")
                results[ctrl]["max_x"].append(np.nan)
                results[ctrl]["crossed"].append(False)

    # Dual-panel: (left) max x bars grouped by height, (right) crossed-fraction
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5),
                            gridspec_kw={"width_ratios": [2.2, 1]})
    n_h = len(heights_mm)
    n_ctrl = len(CTRL_ORDER)
    width = 0.16
    x = np.arange(n_h)
    for i, ctrl in enumerate(CTRL_ORDER):
        offsets = x + (i - (n_ctrl - 1) / 2) * width
        axs[0].bar(offsets, results[ctrl]["max_x"], width,
                   label=LABELS[ctrl], color=COLORS[ctrl])
        # ✓ annotation on crossed
        for hi, crossed in enumerate(results[ctrl]["crossed"]):
            if crossed:
                axs[0].text(x[hi] + (i - (n_ctrl - 1) / 2) * width,
                            results[ctrl]["max_x"][hi] + 0.8,
                            "✓", ha="center", va="bottom",
                            fontsize=10, fontweight="bold")
    axs[0].axhline(11.0, linestyle="--", color="black", linewidth=0.8,
                   label="flight end (x=11 cm)")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"h = {h} mm" for h in heights_mm])
    axs[0].set_ylabel("max forward x (cm)")
    axs[0].set_title("(a) 3-step-flight reach as step height varies")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3, axis="y")

    # Right panel: heatmap-style crossing indicator
    crossed_matrix = np.array([results[c]["crossed"] for c in CTRL_ORDER], dtype=float)
    axs[1].imshow(crossed_matrix, aspect="auto", cmap="RdYlGn",
                  vmin=0, vmax=1, interpolation="nearest")
    axs[1].set_xticks(range(n_h))
    axs[1].set_xticklabels([f"{h}mm" for h in heights_mm])
    axs[1].set_yticks(range(n_ctrl))
    axs[1].set_yticklabels([LABELS[c] for c in CTRL_ORDER])
    for i in range(n_ctrl):
        for j in range(n_h):
            axs[1].text(j, i, "✓" if crossed_matrix[i, j] else "✗",
                        ha="center", va="center", color="white",
                        fontsize=13, fontweight="bold")
    axs[1].set_title("(b) Crossed full flight?")
    axs[1].set_xlabel("step height")

    fig.suptitle("Robustness across step heights — 3-step flight, 5 controllers",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "flight_height_sweep.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (k) Dynamics-model training curves (MSE)
# ---------------------------------------------------------------------------

DYN_CURVE_RUNS = [
    ("hopping_curves_v1",        "7 cm data only",   "#b6b6b6"),
    ("hopping_curves_combined",  "4 + 7 + 10 cm hw", "#8a8a8a"),
    ("hopping_curves_simaug_v1", "hw + sim_v1",      "#ff7f0e"),
    ("hopping_curves_simaug_v2", "hw + sim_combined","#2ca02c"),
]


def fig_k_dynamics_training_curves():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for run, label, color in DYN_CURVE_RUNS:
        curves_path = f"experiments/weights/{run}_curves.npz"
        try:
            d = np.load(curves_path)
        except Exception as e:
            print(f"  (skipping {run}: {e})")
            continue
        tr = d["train_losses"]; va = d["val_losses"]
        epochs = np.arange(1, len(tr) + 1)
        axs[0].plot(epochs, tr, "-",  lw=1.4, color=color, label=f"{label} (train)")
        axs[0].plot(epochs, va, "--", lw=1.4, color=color, alpha=0.85)
        axs[1].plot(epochs, tr, "-", lw=1.4, color=color, label=f"{label} (train)")
        axs[1].plot(epochs, va, "--", lw=1.4, color=color, alpha=0.85)
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE on normalised targets")
        ax.grid(alpha=0.3)
    axs[0].set_title("(a) Dynamics-MLP training — linear scale")
    axs[1].set_title("(b) Same curves, log-y")
    axs[1].set_yscale("log")
    # Dashed lines used for val; put both in a single legend
    from matplotlib.lines import Line2D
    custom = []
    for _, label, color in DYN_CURVE_RUNS:
        custom.append(Line2D([0], [0], color=color, lw=2, label=label))
    custom.append(Line2D([0], [0], color="black", lw=1, linestyle="-", label="train"))
    custom.append(Line2D([0], [0], color="black", lw=1, linestyle="--", label="val"))
    axs[0].legend(handles=custom, fontsize=8, loc="upper right")

    fig.suptitle("HopperMLP dynamics model — training curves across data configs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "dyn_training_curves.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (l) Rollout prediction: teacher-forced vs autonomous
# ---------------------------------------------------------------------------

def fig_l_rollout_prediction(run_name="hopping_curves_simaug_v2", horizon=80):
    """Take a trained dynamics model + held-out test data, roll out in both
    modes, plot per-dim state trajectories."""
    from hopper.models import HopperMLP
    from hopper.data_utils import load_jumping_data

    weights_path = f"experiments/weights/{run_name}.pt"
    norm_path = f"experiments/weights/{run_name}_norm.npz"

    # Load + normalise
    norm = np.load(norm_path)
    X_mean = norm["X_mean"]; X_std = norm["X_std"]
    Y_mean = norm["Y_mean"]; Y_std = norm["Y_std"]

    model = HopperMLP(input_dim=14, output_dim=6, hidden_dim=32)
    model.load(weights_path, device="cpu")
    model.eval()

    # Reproduce the same temporal split as scripts/train.py (last 20% = test)
    data = load_jumping_data(
        "/Users/cassandrahe/MIT Dropbox/Cassandra He/jumping_data"
    )
    X = data.X.astype(np.float32); Y = data.Y.astype(np.float32)
    n_test = int(0.20 * len(X))
    X_test = X[-n_test:]; Y_test = Y[-n_test:]

    # Starting state = first state of the test window
    state0 = X_test[0, :6].copy()

    # --- teacher-forced: predict Δ from TRUE input at each step ---
    tf_states = [state0.copy()]
    # autonomous: predict Δ from PREDICTED state + recorded action at each step
    auto_state = state0.copy()
    auto_states = [auto_state.copy()]

    for t in range(min(horizon, len(X_test) - 1)):
        x_true = X_test[t]
        x_tf_n = (x_true - X_mean) / (X_std + 1e-8)
        with torch.no_grad():
            dy_n = model(torch.tensor(x_tf_n, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        dy = dy_n * (Y_std + 1e-8) + Y_mean
        tf_states.append(tf_states[-1] + dy)

        # autonomous: use predicted state's first 6 dims, keep original action part
        x_auto = x_true.copy()
        x_auto[:6] = auto_state  # swap in predicted state
        x_auto_n = (x_auto - X_mean) / (X_std + 1e-8)
        with torch.no_grad():
            dy_n = model(torch.tensor(x_auto_n, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        dy = dy_n * (Y_std + 1e-8) + Y_mean
        auto_state = auto_state + dy
        auto_states.append(auto_state.copy())

    tf_states = np.array(tf_states)
    auto_states = np.array(auto_states)

    # Ground truth is the same integration using Y_test deltas (which correspond
    # to the recorded inputs X_test[t]).
    gt = [state0.copy()]
    for t in range(min(horizon, len(X_test) - 1)):
        gt.append(gt[-1] + Y_test[t])
    gt = np.array(gt)

    labels = ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"]
    fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axs = axs.ravel()
    for i in range(6):
        ax = axs[i]
        ax.plot(gt[:, i], color="black", lw=1.8, label="ground truth")
        ax.plot(tf_states[:, i], color="#1f77b4", lw=1.2, linestyle="--",
                label="teacher-forced")
        ax.plot(auto_states[:, i], color="#d62728", lw=1.2, linestyle=":",
                label="autonomous")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8, loc="upper right")
    axs[4].set_xlabel("step (0.01 s each)")
    axs[5].set_xlabel("step (0.01 s each)")
    fig.suptitle(f"Rollout-prediction error on held-out test data  ({run_name})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "rollout_prediction.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (m) MPC height tracking / failure modes
# ---------------------------------------------------------------------------

def fig_m_mpc_failure_modes():
    """CEM+NN MPC for each of four dynamics models, averaged over N seeds.

    The dynamics models were all trained on the pre-physics-fix XML, so
    this figure evaluates them on `assets/hopper_prefix.xml` — the same
    regime as progress.md §3.3. CEM is stochastic, so we average z(t)
    across 5 RNG seeds and report peak mean ± std."""
    from hopper.models import HopperMLP
    from hopper.mpc import cem_optimize, set_normalization_stats
    from hopper.frame_alignment import build_R_match, get_body_rotation_matrix
    from hopper.simulation import get_body_pos_eul
    from scipy.spatial.transform import Rotation as Rot

    XML = "assets/hopper_prefix.xml"
    N_SEEDS = 5
    Z_DES = 0.08

    def run_one(run_name, seed, sim_time=3.0, fps=100, z_des=Z_DES):
        weights = f"experiments/weights/{run_name}.pt"
        norm = np.load(f"experiments/weights/{run_name}_norm.npz")
        set_normalization_stats(norm["X_mean"], norm["X_std"],
                                norm["Y_mean"], norm["Y_std"])
        net = HopperMLP(input_dim=14, output_dim=6, hidden_dim=32)
        net.load(weights, device="cpu"); net.eval()
        np.random.seed(seed)
        m = mujoco.MjModel.from_xml_path(XML)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
        spf = max(1, int(round((1.0/fps) / m.opt.timestep)))
        bid = body_name_to_id(m, "hopper")
        zs = []; us = []
        R_ref = np.array([1, 0, 0], dtype=np.float32)
        for i in range(int(sim_time * fps)):
            pos, eul = get_body_pos_eul(m, d, body_name="hopper")
            R_body = get_body_rotation_matrix(m, d, body_name="hopper")
            R_match = build_R_match(R_body, R_ref)
            R_nn = R_match.T @ R_body
            eul_nn = Rot.from_matrix(R_nn).as_euler("xyz").astype(np.float32)
            u0 = cem_optimize(net, pos, eul_nn, z_des=z_des, device="cpu", verbose=False)
            for j, aid in enumerate(aids): d.ctrl[aid] = float(u0[j])
            for _ in range(spf): mujoco.mj_step(m, d)
            zs.append(d.xpos[bid][2]); us.append(np.sum(u0))
        return (np.arange(len(zs)) / fps, np.array(zs), np.array(us))

    runs = [
        ("hopping_v1",           "7 cm data only",   "#888888"),
        ("hopping_combined_v1",  "hw combined",      "#444444"),
        ("hopping_simaug_v1",    "hw + sim_v1",      "#ff7f0e"),
        ("hopping_simaug_v2",    "hw + sim_combined","#2ca02c"),
    ]

    # Aggregate multiple CEM seeds: each run_name → N_SEEDS z(t) arrays.
    results = {}
    for run, _, _ in runs:
        results[run] = {"zs": [], "us": [], "t": None, "peaks": []}
        for seed in range(N_SEEDS):
            t, z, u = run_one(run, seed=seed)
            results[run]["zs"].append(z)
            results[run]["us"].append(u)
            results[run]["t"] = t
            results[run]["peaks"].append(z.max() * 100)
        peaks = np.array(results[run]["peaks"])
        print(f"  {run}: peak z = {peaks.mean():.2f} ± {peaks.std():.2f} cm "
              f"across {N_SEEDS} seeds")

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for run, label, color in runs:
        zs = np.array(results[run]["zs"])
        us = np.array(results[run]["us"])
        t  = results[run]["t"]
        z_mean = zs.mean(axis=0); z_std = zs.std(axis=0)
        peak_mean = float(np.mean(results[run]["peaks"]))
        peak_std  = float(np.std(results[run]["peaks"]))
        axs[0].plot(t, z_mean * 100, lw=1.6, color=color,
                    label=f"{label} — peak {peak_mean:.1f} ± {peak_std:.1f} cm")
        axs[0].fill_between(t, (z_mean - z_std) * 100, (z_mean + z_std) * 100,
                            color=color, alpha=0.18)
        axs[1].plot(t, us.mean(axis=0) * 1000, lw=1.3, color=color)

    axs[0].axhline(Z_DES * 100, color="black", linestyle="--", lw=0.9,
                   label=f"z_des = {Z_DES*100:.0f} cm")
    axs[0].axhline(9.6, color="grey", linestyle=":", lw=0.7,
                   label="hover thrust threshold (9.6 mN)")
    axs[0].set_ylabel("torso z (cm)")
    axs[0].set_title("(a) MPC closed-loop height — four dynamics models, four failure modes")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.25)

    axs[1].axhline(9.6, color="grey", linestyle=":", lw=0.7)
    axs[1].set_ylabel("total thrust (mN)")
    axs[1].set_xlabel("time (s)")
    axs[1].set_title("(b) CEM-commanded total thrust")
    axs[1].grid(alpha=0.25)

    fig.suptitle("CEM + NN-dynamics MPC on the hopper — training-data regime "
                 f"determines the failure (mean ± σ over {N_SEEDS} CEM seeds, "
                 "pre-fix XML)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "mpc_failure_modes.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (n) IL v1 vs v2 drift comparison
# ---------------------------------------------------------------------------

def _extract_il_v1_state_7dim(m, d, z_des):
    """Re-implementation of the original 7-dim IL state (pre-translation-invariant).
    Matches what il_v1 was trained with:
        [z, vz, roll, pitch, omega_roll, omega_pitch, z_des]
    """
    from hopper.simulation import get_body_pos_eul
    pos, eul = get_body_pos_eul(m, d, body_name="hopper")
    vz = float(d.qvel[2])
    wr = float(d.qvel[3]); wp = float(d.qvel[4])
    return np.array([pos[2], vz, eul[0], eul[1], wr, wp, z_des], dtype=np.float32)


def fig_n_il_v1_v2_drift():
    """Run il_v1 (7-dim, hover-only demos) and il_v2 (11-dim, + x, y inputs)
    on the same hover task and plot the xy trajectory.

    Both policies were trained on the pre-physics-fix hopper.xml (before
    the slider-limit and contact-damping fixes documented in §22). To
    reproduce the §6 closed-loop numbers faithfully we evaluate them on
    that same pre-fix XML — `assets/hopper_prefix.xml` — rather than on
    the corrected physics."""
    from hopper.il_policy import ILPolicy

    XML = "assets/hopper_prefix.xml"

    def run_v1(sim_time=3.0):
        norm = np.load("experiments/weights/il_v1_norm.npz")
        X_mean = norm["X_mean"].astype(np.float32); X_std = norm["X_std"].astype(np.float32)
        policy = ILPolicy(input_dim=7, output_dim=4, hidden_dim=64)
        policy.load("experiments/weights/il_v1.pt"); policy.eval()
        m = mujoco.MjModel.from_xml_path(XML)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
        spf = max(1, int(round(0.01 / m.opt.timestep)))
        bid = body_name_to_id(m, "hopper")
        xs, ys, zs = [], [], []
        Z_DES = 0.08
        for i in range(int(sim_time * 100)):
            s = _extract_il_v1_state_7dim(m, d, Z_DES)
            s_n = (s - X_mean) / X_std
            with torch.no_grad():
                u = policy(torch.tensor(s_n).unsqueeze(0)).numpy()[0]
            for j, aid in enumerate(aids): d.ctrl[aid] = float(u[j])
            for _ in range(spf): mujoco.mj_step(m, d)
            p = d.xpos[bid]; xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        return (np.arange(len(xs)) * 0.01, np.array(xs), np.array(ys), np.array(zs))

    def run_v2(sim_time=3.0):
        from hopper.il_policy import ILPolicy as IL2
        norm = np.load("experiments/weights/il_v2_norm.npz")
        X_mean = norm["X_mean"].astype(np.float32); X_std = norm["X_std"].astype(np.float32)
        policy = IL2(input_dim=11, output_dim=4, hidden_dim=64)
        policy.load("experiments/weights/il_v2.pt"); policy.eval()
        m = mujoco.MjModel.from_xml_path(XML)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
        spf = max(1, int(round(0.01 / m.opt.timestep)))
        bid = body_name_to_id(m, "hopper")
        xs, ys, zs = [], [], []
        Z_DES = 0.08
        for i in range(int(sim_time * 100)):
            # Pre-translation-invariant: il_v2 used absolute x, y → call with x_des=0
            s = extract_il_state(m, d, Z_DES, x_des=0.0, y_des=0.0)
            s_n = (s - X_mean) / X_std
            with torch.no_grad():
                u = policy(torch.tensor(s_n).unsqueeze(0)).numpy()[0]
            for j, aid in enumerate(aids): d.ctrl[aid] = float(u[j])
            for _ in range(spf): mujoco.mj_step(m, d)
            p = d.xpos[bid]; xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        return (np.arange(len(xs)) * 0.01, np.array(xs), np.array(ys), np.array(zs))

    t1, x1, y1, z1 = run_v1()
    t2, x2, y2, z2 = run_v2()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    # Top-down xy trajectory
    axs[0].plot(x1 * 100, y1 * 100, "-", lw=1.5, color="#d62728",
                label=f"il_v1 (7-dim, no x/y)  drift={np.hypot(x1[-1], y1[-1])*100:.1f} cm")
    axs[0].plot(x2 * 100, y2 * 100, "-", lw=1.5, color="#1f77b4",
                label=f"il_v2 (11-dim, has x/y)  drift={np.hypot(x2[-1], y2[-1])*100:.1f} cm")
    axs[0].scatter([0], [0], s=80, marker="o", color="black", zorder=5,
                   edgecolors="white", label="start (0, 0)")
    axs[0].scatter([x1[-1]*100], [y1[-1]*100], s=60, marker="X", color="#d62728",
                   zorder=5, edgecolors="white")
    axs[0].scatter([x2[-1]*100], [y2[-1]*100], s=60, marker="X", color="#1f77b4",
                   zorder=5, edgecolors="white")
    axs[0].set_xlabel("x (cm)")
    axs[0].set_ylabel("y (cm)")
    axs[0].set_title("(a) top-down xy drift over 3 s")
    axs[0].set_aspect("equal")
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper left")
    axs[0].axhline(0, color="grey", lw=0.5)
    axs[0].axvline(0, color="grey", lw=0.5)

    # z tracking
    axs[1].plot(t1, z1 * 100, lw=1.2, color="#d62728", label="il_v1")
    axs[1].plot(t2, z2 * 100, lw=1.2, color="#1f77b4", label="il_v2")
    axs[1].axhline(8.0, color="black", linestyle="--", lw=0.8, label="z_des = 8 cm")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("torso z (cm)")
    axs[1].set_title("(b) altitude tracking (both policies hover)")
    axs[1].legend(fontsize=8, loc="lower right")
    axs[1].grid(alpha=0.3)

    fig.suptitle("Effect of including x, y in the IL state — hover at z_des = 8 cm",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "il_v1_v2_drift.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (o) Energy vs bang-bang thrust profile
# ---------------------------------------------------------------------------

def fig_o_energy_vs_bangbang():
    """Run the two hopping experts side by side on the same fixed-XML scene
    and plot total thrust over time to visualise how the two expert styles
    differ."""
    from hopper.sim_data_collection import (
        PDGains, HopGains, HoppingState, EnergyHopGains,
        hopping_control, energy_hopping_control,
    )

    def run_expert(mode, sim_time=2.0, z_des=0.08):
        m = mujoco.MjModel.from_xml_path("assets/hopper.xml")
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        mass = float(np.sum(m.body_mass)); g = float(-m.opt.gravity[2])
        aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
        spf = max(1, int(round(0.01 / m.opt.timestep)))
        bid = body_name_to_id(m, "hopper")

        att = PDGains()
        if mode == "bangbang":
            hop = HopGains(z_switch_frac=0.6)
            state = HoppingState(z_des)
        else:
            eg = EnergyHopGains()

        zs, Fs = [], []
        for i in range(int(sim_time * 100)):
            if mode == "bangbang":
                u = hopping_control(m, d, z_des, mass, g, 0.015, 0.003, att, hop, state)
            else:
                u = energy_hopping_control(m, d, z_des, mass, g, 0.015, 0.003, att, eg)
            for j, aid in enumerate(aids): d.ctrl[aid] = float(u[j])
            for _ in range(spf): mujoco.mj_step(m, d)
            zs.append(d.xpos[bid][2]); Fs.append(float(np.sum(u)))
        return (np.arange(len(zs)) * 0.01, np.array(zs), np.array(Fs))

    t_bb, z_bb, F_bb = run_expert("bangbang")
    t_en, z_en, F_en = run_expert("energy")

    # Three panels: full 2 s trajectory; zoomed single cycle; thrust scatter
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1],
                          hspace=0.45, wspace=0.25)
    ax_z = fig.add_subplot(gs[0, :])
    ax_F = fig.add_subplot(gs[1, :], sharex=ax_z)
    ax_zm_z = fig.add_subplot(gs[2, 0])
    ax_zm_F = fig.add_subplot(gs[2, 1])

    ax_z.plot(t_bb, z_bb * 100, lw=1.4, color="#d62728", label="bang-bang (hopping_control)")
    ax_z.plot(t_en, z_en * 100, lw=1.4, color="#2ca02c", label="energy-shaping (energy_hopping_control)")
    ax_z.axhline(8.0, color="black", linestyle="--", lw=0.8, label="z_des = 8 cm")
    ax_z.set_ylabel("torso z (cm)")
    ax_z.set_title("(a) z over 2 s")
    ax_z.legend(fontsize=9, loc="upper right")
    ax_z.grid(alpha=0.3)

    ax_F.plot(t_bb, F_bb * 1000, lw=1.4, color="#d62728", label="bang-bang")
    ax_F.plot(t_en, F_en * 1000, lw=1.4, color="#2ca02c", label="energy-shaping")
    ax_F.axhline(9.6, color="grey", linestyle=":", lw=0.8, label="hover = 9.6 mN")
    ax_F.axhline(12.0, color="grey", linestyle="-.", lw=0.8, label="F_max = 12 mN")
    ax_F.set_ylabel("total thrust F (mN)")
    ax_F.set_xlabel("time (s)")
    ax_F.set_title("(b) total thrust command")
    ax_F.legend(fontsize=9, loc="upper right")
    ax_F.grid(alpha=0.3)

    # Zoomed single cycle (0.30 – 0.55 s captures one pump-then-coast phase)
    zw_lo, zw_hi = 0.30, 0.55
    mask_bb = (t_bb >= zw_lo) & (t_bb <= zw_hi)
    mask_en = (t_en >= zw_lo) & (t_en <= zw_hi)
    ax_zm_z.plot(t_bb[mask_bb], z_bb[mask_bb] * 100, lw=1.4, color="#d62728",
                 label="bang-bang")
    ax_zm_z.plot(t_en[mask_en], z_en[mask_en] * 100, lw=1.4, color="#2ca02c",
                 label="energy-shaping")
    ax_zm_z.axhline(8.0, color="black", linestyle="--", lw=0.8)
    ax_zm_z.set_xlabel("time (s)")
    ax_zm_z.set_ylabel("z (cm)")
    ax_zm_z.set_title("(c) zoom on one hop — z")
    ax_zm_z.grid(alpha=0.3)

    ax_zm_F.step(t_bb[mask_bb], F_bb[mask_bb] * 1000, where="post", lw=1.4,
                 color="#d62728", label="bang-bang (steps ↑0 or 12 mN)")
    ax_zm_F.plot(t_en[mask_en], F_en[mask_en] * 1000, lw=1.4, color="#2ca02c",
                 label="energy-shaping (continuous)")
    ax_zm_F.axhline(9.6, color="grey", linestyle=":", lw=0.8)
    ax_zm_F.axhline(12.0, color="grey", linestyle="-.", lw=0.8)
    ax_zm_F.set_xlabel("time (s)")
    ax_zm_F.set_ylabel("F (mN)")
    ax_zm_F.set_title("(d) zoom on one hop — thrust profile")
    ax_zm_F.grid(alpha=0.3)
    ax_zm_F.legend(fontsize=8, loc="upper right")

    fig.suptitle("Two hopping experts: bang-bang pump-with-apex-feedback vs "
                 "smooth energy shaping", fontsize=12, fontweight="bold")
    out = os.path.join(OUT_DIR, "expert_thrust_profiles.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (p) Domain-randomisation sim-to-real sweep
# ---------------------------------------------------------------------------

def fig_p_dr_sweep():
    """Two-panel heatmap of max-x reached on the 3-step flight across a
    grid of (contact-damping, wing-gain) multipliers, for `ppo_mix_v1`
    (trained on nominal physics) and `ppo_mix_dr_v1` (trained with DR).
    The robustness claim: DR-trained should degrade more gracefully as
    physics drift from nominal."""
    data = np.load("experiments/ppo/dr_sweep_results.npz")
    dampings = data["dampings"]; gains = data["gains"]
    nom = data["max_x_nominal"]; dr = data["max_x_dr"]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3),
                            gridspec_kw={"width_ratios": [1, 1, 1]})

    # Share colour scale between the two max-x heatmaps so the comparison is
    # visually consistent. Flight end at 11 cm is the "crossed" threshold.
    vmin = min(np.nanmin(nom), np.nanmin(dr))
    vmax = max(np.nanmax(nom), np.nanmax(dr))

    for ax, matrix, title in [
        (axs[0], nom, "ppo_mix_v1 (nominal training)"),
        (axs[1], dr,  "ppo_mix_dr_v1 (DR training)"),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       origin="lower")
        ax.set_xticks(range(len(gains)))
        ax.set_xticklabels([f"{g:.2f}" for g in gains], fontsize=9)
        ax.set_yticks(range(len(dampings)))
        ax.set_yticklabels([f"{d:.2f}" for d in dampings], fontsize=9)
        ax.set_xlabel("wing-gain multiplier")
        ax.set_ylabel("contact-damping multiplier")
        ax.set_title(title, fontsize=11)
        # Annotate crossed cells with ✓
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v): continue
                txt = "✓" if v > 11 else "✗"
                col = "white" if v > 11 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=col, fontsize=10, fontweight="bold")
        # Highlight nominal training point (1.0, 1.0) with a border
        i_nom = np.argmin(np.abs(dampings - 1.0))
        j_nom = np.argmin(np.abs(gains - 1.0))
        ax.add_patch(plt.Rectangle((j_nom - 0.5, i_nom - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=2))

    fig.colorbar(im, ax=axs[:2].tolist(), shrink=0.7,
                 label="max x reached (cm)")

    # Third panel: crossing-success aggregated as a function of damping
    # multiplier (marginalise over gain).
    ax = axs[2]
    nom_cross = (nom > 11).mean(axis=1) * 100
    dr_cross  = (dr  > 11).mean(axis=1) * 100
    ax.plot(dampings, nom_cross, "o-", lw=1.8, color=COLORS["ppo_mix_v1"],
            label=LABELS["ppo_mix_v1"])
    ax.plot(dampings, dr_cross, "s-", lw=1.8, color="#9467bd",
            label="ppo_mix_dr_v1 (DR)")
    ax.set_xlabel("contact-damping multiplier")
    ax.set_ylabel("% of gain configs crossed")
    ax.set_title("crossing rate vs damping")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 105)

    fig.suptitle("Domain-randomisation robustness — climb under perturbed physics",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "dr_sweep.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# (q) IL-init vs random-init PPO
# ---------------------------------------------------------------------------

def fig_q_ilinit_vs_random():
    """Compare ppo_mix_v1 (random-init) vs ppo_mix_ilinit_v1 (naive IL-init)
    vs ppo_mix_ilkl_v1 (KL-regularised IL-init) on two axes: (a) learning
    curves from tensorboard, (b) closed-loop performance on the 4 standard
    scenarios."""
    scenarios = [
        Scenario("flat hover",       "assets/hopper.xml", 0.08, 0.00,  0.00),
        Scenario("flat fwd",         "assets/hopper.xml", 0.08, 0.03, -0.03),
        Scenario("single step",      "assets/hopper_stair_h8_d20.xml", 0.08, 0.03, -0.05),
        Scenario("3-step flight",    "assets/hopper_stair_flight_3x8mm.xml", 0.10, 0.03, -0.05, 7.0),
    ]

    # Closed-loop eval on all 4 scenarios for three PPO variants
    runs_to_eval = [
        ("ppo_mix_v1",        "random-init",        COLORS["ppo_mix_v1"]),
        ("ppo_mix_ilinit_v1", "naive IL-init",      "#9467bd"),
        ("ppo_mix_ilkl_v1",   "KL-reg IL-init",     "#e377c2"),
    ]

    cl_data = {scn.name: {} for scn in scenarios}
    for scn in scenarios:
        for run_name, label, _ in runs_to_eval:
            _, xs, _, _ = rollout_ppo(run_name, scn)
            cl_data[scn.name][run_name] = (float(xs.max() * 100), len(xs))

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.3)
    ax_rew = fig.add_subplot(gs[0, 0])
    ax_len = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    # Top row: learning curves
    for run_name, label, color in runs_to_eval:
        try:
            s, v = _load_tb(run_name, "rollout/ep_rew_mean")
            ax_rew.plot(s / 1000, v, lw=1.5, color=color, label=label)
            s, v = _load_tb(run_name, "rollout/ep_len_mean")
            ax_len.plot(s / 1000, v, lw=1.5, color=color, label=label)
        except Exception as e:
            print(f"  (skipping tb for {run_name}: {e})")
    ax_rew.set_xlabel("timesteps (×10³)")
    ax_rew.set_ylabel("mean episode reward")
    ax_rew.set_title("(a) reward vs training steps")
    ax_rew.legend(fontsize=9); ax_rew.grid(alpha=0.3)
    ax_len.set_xlabel("timesteps (×10³)")
    ax_len.set_ylabel("mean episode length")
    ax_len.set_title("(b) episode length vs training steps")
    ax_len.legend(fontsize=9); ax_len.grid(alpha=0.3)

    # Bottom row: closed-loop max-x bars
    n_scn = len(scenarios)
    n_run = len(runs_to_eval)
    width = 0.8 / n_run
    x = np.arange(n_scn)
    for i, (run_name, label, color) in enumerate(runs_to_eval):
        offsets = x + (i - (n_run - 1) / 2) * width
        max_xs = [cl_data[scn.name][run_name][0] for scn in scenarios]
        ax_bar.bar(offsets, max_xs, width, label=label, color=color)
        # ✓ on crossed-flight
        for si, scn in enumerate(scenarios):
            if scn.name == "3-step flight" and cl_data[scn.name][run_name][0] > 11:
                ax_bar.text(x[si] + (i - (n_run - 1) / 2) * width,
                            cl_data[scn.name][run_name][0] + 1,
                            "✓", ha="center", fontsize=11, fontweight="bold")
    ax_bar.axhline(11, color="black", linestyle="--", lw=0.8,
                   label="flight end (x=11 cm)")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([s.name for s in scenarios])
    ax_bar.set_ylabel("max forward x (cm)")
    ax_bar.set_yscale("symlog")
    ax_bar.set_title("(c) closed-loop max forward distance across 4 scenarios")
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.grid(alpha=0.3, axis="y")

    fig.suptitle("IL warm-start variants vs random init — ppo_mix "
                 "(1 M steps, 128², flat + stair)",
                 fontsize=12, fontweight="bold")
    out = os.path.join(OUT_DIR, "ilinit_vs_random.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    only = sys.argv[1:]   # pass e.g. 'h' to regenerate one

    figs = {
        "a": fig_a_bar_comparison,
        "b": fig_b_com_trajectory,
        "c": fig_c_learning_curves,
        "d": fig_d_hop_cycles,
        "e": fig_e_stair_schematic,
        "f": fig_f_per_wing_forces,
        "g": fig_g_apex_histogram,
        "h": fig_h_multi_seed,
        "i": fig_i_stair_height_sweep,
        "j": fig_j_flight_height_sweep,
        "k": fig_k_dynamics_training_curves,
        "l": fig_l_rollout_prediction,
        "m": fig_m_mpc_failure_modes,
        "n": fig_n_il_v1_v2_drift,
        "o": fig_o_energy_vs_bangbang,
        "p": fig_p_dr_sweep,
        "q": fig_q_ilinit_vs_random,
    }
    targets = only if only else list(figs)
    print(f"Writing figures to {OUT_DIR}/  (targets: {targets})")
    for t in targets:
        if t in figs:
            figs[t]()
    print("Done.")
