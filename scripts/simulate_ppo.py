"""
Closed-loop evaluation of a trained PPO policy.

Parallels scripts/simulate_il.py: same --xml / --x0 / --y0 / --z0 /
--z-des / --vx-des interface so an RL policy can be dropped onto the
same eval harness as the IL ones. Also dumps an .mp4 of the rollout.
"""

import argparse
import os

import numpy as np
import mujoco
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hopper.rl_env import HopperEnv, HopperEnvConfig
from hopper.simulation import body_name_to_id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ppo_v1")
    p.add_argument("--outdir", default="experiments/ppo")
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--sim-time", type=float, default=5.0)
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--z-des", type=float, default=0.08)
    p.add_argument("--vx-des", type=float, default=0.0)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=0.035)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    rundir = os.path.join(args.outdir, args.run_name)
    model_path = os.path.join(rundir, "model.zip")
    norm_path = os.path.join(rundir, "vec_normalize.pkl")

    cfg = HopperEnvConfig(
        xml_path=args.xml,
        max_episode_seconds=args.sim_time,
        z_des=args.z_des, vx_des=args.vx_des,
        randomize_task=False,
        z_init_lo=args.z0, z_init_hi=args.z0,
        xy_range_init=0.0, tilt_deg_init=0.0,
    )

    def _factory():
        e = HopperEnv(cfg)
        # Enforce custom initial x, y (reset randomises within xy_range, but
        # we set range=0 above so start is always (x0, y0, z0)).
        e.reset()
        e.data.qpos[0] = args.x0
        e.data.qpos[1] = args.y0
        e.data.qpos[2] = args.z0
        e.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(e.model, e.data)
        return e

    venv = DummyVecEnv([_factory])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(model_path, env=venv, device="cpu")

    # Pull the raw env for rendering
    env = venv.envs[0].unwrapped
    renderer = mujoco.Renderer(env.model, width=640, height=480)
    bid = body_name_to_id(env.model, "hopper")

    obs = venv.reset()
    # Manually re-initialise so we start at the requested state (DummyVecEnv
    # reset calls env.reset() which randomises internally based on our cfg)
    env.data.qpos[0] = args.x0
    env.data.qpos[1] = args.y0
    env.data.qpos[2] = args.z0
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    env._step_count = 0
    env._prev_x = args.x0
    env._z_des = args.z_des
    env._vx_des = args.vx_des
    mujoco.mj_forward(env.model, env.data)

    n = int(args.sim_time * args.fps)
    xs, ys, zs, frames, rewards = [], [], [], [], []
    for i in range(n):
        # Log position BEFORE step — after a terminating step, VecEnv
        # auto-resets and env.data reflects the reset state, not the
        # terminal state.
        xs.append(env.data.xpos[bid][0])
        ys.append(env.data.xpos[bid][1])
        zs.append(env.data.xpos[bid][2])
        if not args.no_video:
            renderer.update_scene(env.data)
            frames.append(renderer.render())

        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, info = venv.step(action)
        rewards.append(float(reward[0]))
        if done[0]:
            # Log terminal state before the auto-reset overwrites env.data.
            # info[0]["terminal_observation"] has the obs, but we want xpos;
            # the most recent xpos (already appended above) is the pre-step
            # state — log one more "post-step, pre-reset" by peeking at the
            # info dict's x/z that the env stashed.
            if "x" in info[0]:
                xs.append(info[0]["x"])
                zs.append(info[0]["z"])
                ys.append(ys[-1])  # y not in info; use last known
            print(f"episode ended at step {i}")
            break

    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    print(f"run: {args.run_name}  z_des={args.z_des*100:.1f}cm vx_des={args.vx_des*100:.1f}cm/s")
    print(f"  total reward: {sum(rewards):.1f}  mean/step: {np.mean(rewards):.3f}  steps: {len(rewards)}")
    print(f"  z mean/std: {zs.mean()*100:.2f}/{zs.std()*100:.2f} cm   z_max: {zs.max()*100:.2f}")
    print(f"  final pos: x={xs[-1]*100:+.2f} y={ys[-1]*100:+.2f} z={zs[-1]*100:+.2f}")
    print(f"  drift: {np.hypot(xs[-1] - args.x0, ys[-1] - args.y0)*100:.2f} cm from start")

    if not args.no_video:
        vid_dir = os.path.join(rundir, "videos")
        os.makedirs(vid_dir, exist_ok=True)
        vid = os.path.join(vid_dir, f"eval_z{int(args.z_des*100)}_vx{int(args.vx_des*100)}_x{int(args.x0*100)}.mp4")
        imageio.mimsave(vid, frames, fps=args.fps)
        print(f"  video → {vid}")


if __name__ == "__main__":
    main()
