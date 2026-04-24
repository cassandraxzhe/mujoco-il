"""
Sim-to-real robustness sweep: evaluate `ppo_mix_v1` (trained on nominal
physics) and `ppo_mix_dr_v1` (trained with domain randomisation) across
a grid of perturbed physical parameters. For each (damping multiplier,
wing-gain multiplier) combination, re-run both policies on the 4-step
flight and record max forward x (climb progress) + episode length
(stability).

Saves:
    experiments/ppo/dr_sweep_results.npz  with keys:
        dampings (K,)        — damping multipliers tested
        gains    (K,)        — wing-gain multipliers tested
        max_x_nominal   (len(d), len(g))   — ppo_mix_v1 max x (cm)
        max_x_dr        (len(d), len(g))   — ppo_mix_dr_v1 max x (cm)
        steps_nominal   (len(d), len(g))   — ppo_mix_v1 episode length
        steps_dr        (len(d), len(g))   — ppo_mix_dr_v1 episode length
"""

import argparse
import os
from itertools import product

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hopper.rl_env import HopperEnv, HopperEnvConfig
from hopper.simulation import body_name_to_id


def rollout_ppo_perturbed(run_name, damping_mul, gain_mul, xml_path,
                          z_des=0.10, vx_des=0.03, x0=-0.05, sim_time=7.0):
    """Run a trained PPO policy on an env whose physics are scaled by
    (damping_mul, gain_mul) relative to nominal."""
    rundir = os.path.join("experiments/ppo", run_name)
    cfg = HopperEnvConfig(
        xml_path=xml_path, max_episode_seconds=sim_time,
        z_des=z_des, vx_des=vx_des, randomize_task=False,
        z_init_lo=0.035, z_init_hi=0.035,
        xy_range_init=0.0, tilt_deg_init=0.0,
        # Pin DR to the single point we're probing
        dr_damping_range=(damping_mul, damping_mul),
        dr_wing_gain_range=(gain_mul, gain_mul),
    )

    def _factory():
        e = HopperEnv(cfg); e.reset()
        e.data.qpos[0] = x0; e.data.qpos[1] = 0.0; e.data.qpos[2] = 0.035
        e.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(e.model, e.data)
        return e

    venv = DummyVecEnv([_factory])
    venv = VecNormalize.load(os.path.join(rundir, "vec_normalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(os.path.join(rundir, "model.zip"), env=venv, device="cpu")
    env = venv.envs[0].unwrapped
    bid = body_name_to_id(env.model, "hopper")

    obs = venv.reset()
    env.data.qpos[0] = x0; env.data.qpos[1] = 0.0; env.data.qpos[2] = 0.035
    env.data.qpos[3:7] = [1, 0, 0, 0]; env.data.qvel[:] = 0
    env._step_count = 0; env._prev_x = x0
    env._z_des = z_des; env._vx_des = vx_des
    mujoco.mj_forward(env.model, env.data)

    xs = []
    n = int(sim_time * 100)
    for i in range(n):
        xs.append(env.data.xpos[bid][0])
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        if done[0]:
            if "x" in info[0]:
                xs.append(info[0]["x"])
            break
    return np.array(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper_stair_flight_3x8mm.xml")
    p.add_argument("--nominal-run", default="ppo_mix_v1")
    p.add_argument("--dr-run", default="ppo_mix_dr_v1")
    p.add_argument("--out", default="experiments/ppo/dr_sweep_results.npz")
    args = p.parse_args()

    dampings = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
    gains    = np.array([0.7, 0.85, 0.9, 1.0, 1.1, 1.15, 1.3])

    results = {
        "nominal": {"max_x": np.zeros((len(dampings), len(gains))),
                    "steps": np.zeros((len(dampings), len(gains)))},
        "dr":      {"max_x": np.zeros((len(dampings), len(gains))),
                    "steps": np.zeros((len(dampings), len(gains)))},
    }
    for i, d in enumerate(dampings):
        for j, g in enumerate(gains):
            for label, run in [("nominal", args.nominal_run), ("dr", args.dr_run)]:
                try:
                    xs = rollout_ppo_perturbed(run, d, g, args.xml)
                    results[label]["max_x"][i, j] = xs.max() * 100
                    results[label]["steps"][i, j] = len(xs)
                except Exception as e:
                    print(f"  {run} @ (d={d}, g={g}) failed: {e}")
                    results[label]["max_x"][i, j] = np.nan
                    results[label]["steps"][i, j] = np.nan
        print(f"  damping {d:.2f}: done ({(i+1) * len(gains) * 2} total runs)")

    np.savez(args.out,
             dampings=dampings, gains=gains,
             max_x_nominal=results["nominal"]["max_x"],
             max_x_dr=results["dr"]["max_x"],
             steps_nominal=results["nominal"]["steps"],
             steps_dr=results["dr"]["steps"])
    print(f"\n  ✓ {args.out}")

    # Quick summary to stdout
    def summary(label, data):
        mx = data["max_x"]
        print(f"{label:8}  mean max_x = {np.nanmean(mx):6.2f} cm  "
              f"crossed (>11 cm) fraction = {(mx > 11).mean():.2%}")
    print()
    summary("nominal", results["nominal"])
    summary("dr",      results["dr"])


if __name__ == "__main__":
    main()
