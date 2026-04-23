"""
Run the expert, the IL policy, and one or more PPO policies through a
fixed battery of scenarios and print a comparison table.

Used to build the IL-vs-PPO-vs-expert table for the thesis. Each scenario
reports max x reached (locomotion), final z (tipped or not), episode
length (stability), and whether the 3-step flight was crossed.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hopper.rl_env import HopperEnv, HopperEnvConfig, _mix_wings
from hopper.il_policy import (
    ILPolicyFTxTy, extract_il_state, IL_STATE_DIM,
)
from hopper.sim_data_collection import (
    PDGains, EnergyHopGains, energy_hopping_control,
)
from hopper.simulation import body_name_to_id, actuator_name_to_id


@dataclass
class Scenario:
    name: str
    xml: str
    z_des: float
    vx_des: float
    x0: float
    sim_time: float = 5.0


def run_expert(scn: Scenario):
    m = mujoco.MjModel.from_xml_path(scn.xml)
    d = mujoco.MjData(m)
    d.qpos[:3] = [scn.x0, 0.0, 0.035]
    d.qpos[3:7] = [1.0, 0, 0, 0]
    mujoco.mj_forward(m, d)
    mass = float(np.sum(m.body_mass)); g = float(-m.opt.gravity[2])
    att = PDGains(); eg = EnergyHopGains()
    aids = [actuator_name_to_id(m, n) for n in ("f1","f2","f3","f4")]
    spf = max(1, int(round(0.01 / m.opt.timestep)))
    bid = body_name_to_id(m, "hopper")
    xs, ys, zs = [], [], []
    for i in range(int(scn.sim_time * 100)):
        t = i * 0.01
        u = energy_hopping_control(m, d, scn.z_des, mass, g, 0.015, 0.003, att, eg,
                                   x_des=scn.vx_des*t, y_des=0)
        for j, aid in enumerate(aids):
            d.ctrl[aid] = float(u[j])
        for _ in range(spf):
            mujoco.mj_step(m, d)
        p = d.xpos[bid]
        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        # Termination check (tip)
        R = np.asarray(d.xmat).reshape(-1, 3, 3)[bid]
        pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
        roll = np.arctan2(R[2,1], R[2,2])
        if abs(roll) > np.deg2rad(60) or abs(pitch) > np.deg2rad(60):
            break
    return np.array(xs), np.array(ys), np.array(zs)


def run_il(weights: str, norm_npz: str, scn: Scenario):
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
    xs, ys, zs = [], [], []
    for i in range(int(scn.sim_time * 100)):
        t = i * 0.01
        s = extract_il_state(m, d, scn.z_des, x_des=scn.vx_des*t, y_des=0)
        s_n = (s - X_mean) / X_std
        with torch.no_grad():
            u = policy.wing_forces(torch.tensor(s_n).unsqueeze(0)).numpy()[0]
        for j, aid in enumerate(aids):
            d.ctrl[aid] = float(u[j])
        for _ in range(spf):
            mujoco.mj_step(m, d)
        p = d.xpos[bid]
        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        R = np.asarray(d.xmat).reshape(-1, 3, 3)[bid]
        pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
        roll = np.arctan2(R[2,1], R[2,2])
        if abs(roll) > np.deg2rad(60) or abs(pitch) > np.deg2rad(60):
            break
    return np.array(xs), np.array(ys), np.array(zs)


def run_ppo(run_name: str, outdir: str, scn: Scenario):
    model_path = os.path.join(outdir, run_name, "model.zip")
    norm_path = os.path.join(outdir, run_name, "vec_normalize.pkl")
    cfg = HopperEnvConfig(
        xml_path=scn.xml,
        max_episode_seconds=scn.sim_time,
        z_des=scn.z_des, vx_des=scn.vx_des,
        randomize_task=False,
        z_init_lo=0.035, z_init_hi=0.035,
        xy_range_init=0.0, tilt_deg_init=0.0,
    )

    def _factory():
        e = HopperEnv(cfg)
        e.reset()
        e.data.qpos[0] = scn.x0
        e.data.qpos[1] = 0.0
        e.data.qpos[2] = 0.035
        e.data.qpos[3:7] = [1.0, 0, 0, 0]
        mujoco.mj_forward(e.model, e.data)
        return e

    venv = DummyVecEnv([_factory])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_path, env=venv, device="cpu")
    env = venv.envs[0].unwrapped
    bid = body_name_to_id(env.model, "hopper")
    obs = venv.reset()
    env.data.qpos[0] = scn.x0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.035
    env.data.qpos[3:7] = [1.0, 0, 0, 0]
    env.data.qvel[:] = 0.0
    env._step_count = 0; env._prev_x = scn.x0
    env._z_des = scn.z_des; env._vx_des = scn.vx_des
    mujoco.mj_forward(env.model, env.data)

    xs, ys, zs = [], [], []
    for i in range(int(scn.sim_time * 100)):
        xs.append(env.data.xpos[bid][0])
        ys.append(env.data.xpos[bid][1])
        zs.append(env.data.xpos[bid][2])
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        if done[0]:
            if "x" in info[0]:
                xs.append(info[0]["x"])
                zs.append(info[0]["z"])
                ys.append(ys[-1])
            break
    return np.array(xs), np.array(ys), np.array(zs)


def summarize(name: str, xs, ys, zs, scn: Scenario):
    max_x_cm = xs.max() * 100
    final_z_cm = zs[-1] * 100
    # Climbed-all-3-steps: only meaningful on the 3-step flight. Steps end at x=11 cm.
    crossed_flight = (scn.name.startswith("3-step") and xs.max() > 0.11)
    return dict(
        name=name, scenario=scn.name,
        max_x_cm=max_x_cm, final_z_cm=final_z_cm,
        steps=len(xs),
        crossed_flight=crossed_flight,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--il-run", default="il_energy_ftxty_fwd_dag2")
    p.add_argument("--ppo-outdir", default="experiments/ppo")
    p.add_argument("--ppo-runs", nargs="*",
                   default=["ppo_v2", "ppo_stair_v1"])
    args = p.parse_args()

    scenarios = [
        Scenario("flat hover",       "assets/hopper.xml",                    0.08, 0.00,  0.00),
        Scenario("flat fwd vx=3",    "assets/hopper.xml",                    0.08, 0.03, -0.03),
        Scenario("single 8mm step",  "assets/hopper_stair_h8_d20.xml",       0.08, 0.03, -0.05),
        Scenario("3-step flight",    "assets/hopper_stair_flight_3x8mm.xml", 0.10, 0.03, -0.05, 7.0),
    ]

    il_weights = f"experiments/weights/{args.il_run}.pt"
    il_norm = f"experiments/weights/{args.il_run}_norm.npz"

    rows = []
    for scn in scenarios:
        xs, ys, zs = run_expert(scn)
        rows.append(summarize("expert", xs, ys, zs, scn))
        xs, ys, zs = run_il(il_weights, il_norm, scn)
        rows.append(summarize(f"IL  ({args.il_run})", xs, ys, zs, scn))
        for pr in args.ppo_runs:
            xs, ys, zs = run_ppo(pr, args.ppo_outdir, scn)
            rows.append(summarize(f"PPO ({pr})", xs, ys, zs, scn))

    # Print table grouped by scenario
    print(f"\n{'Controller':<40} {'Scenario':<20} {'max_x':>8} {'final_z':>8} {'steps':>6} {'crossed':>8}")
    print("-" * 100)
    current_scn = None
    for row in rows:
        if row["scenario"] != current_scn:
            current_scn = row["scenario"]
            print()
        print(f"{row['name']:<40} {row['scenario']:<20} "
              f"{row['max_x_cm']:>7.2f}cm {row['final_z_cm']:>7.2f}cm "
              f"{row['steps']:>6} {'yes' if row['crossed_flight'] else '-':>8}")


if __name__ == "__main__":
    main()
