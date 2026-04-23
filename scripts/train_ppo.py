"""
PPO training on the hopper environment using Stable-Baselines3.

The env is `HopperEnv` (hopper/rl_env.py), matching the IL pipeline:
  • 11-dim translation-invariant observation
  • 3-dim normalised (F, Tx, Ty) action → analytical mixer → per-wing forces
  • Per-episode task randomisation (z_des ∈ [0.06, 0.10], vx_des ∈ {0} ∪ [0, 0.05])
  • 100 Hz control, 5-second episodes, max 500 steps

Outputs:
    experiments/ppo/<run-name>/
        model.zip             — final model
        checkpoints/*.zip     — periodic checkpoints
        tensorboard/          — logging (launch `tensorboard --logdir …`)
        norm.pkl              — obs normalisation (VecNormalize)
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hopper.rl_env import HopperEnv, HopperEnvConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml",
                   help="Base XML. If --xml-mix is given, splits envs between it and the mix.")
    p.add_argument("--xml-mix", default=None,
                   help="Optional second XML to mix in — half the envs use --xml, half --xml-mix.")
    p.add_argument("--run-name", default="ppo_v1")
    p.add_argument("--outdir", default="experiments/ppo")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=512,
                   help="per-env rollout length (PPO sample buffer size = n_envs × n_steps)")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--device", default="cpu")
    # Env knobs (forwarded to HopperEnvConfig)
    p.add_argument("--z-des-lo", type=float, default=0.06)
    p.add_argument("--z-des-hi", type=float, default=0.10)
    p.add_argument("--vx-des-lo", type=float, default=0.0)
    p.add_argument("--vx-des-hi", type=float, default=0.05)
    p.add_argument("--episode-sec", type=float, default=5.0)
    return p.parse_args()


def make_env_factory(args, rank):
    # Split envs across the two XMLs when --xml-mix is given. Rank-even envs
    # get --xml, rank-odd get --xml-mix. The global PPO rollout sees
    # transitions from both terrains in roughly equal proportion.
    if args.xml_mix and rank % 2 == 1:
        xml = args.xml_mix
    else:
        xml = args.xml

    def _init():
        cfg = HopperEnvConfig(
            xml_path=xml,
            max_episode_seconds=args.episode_sec,
            z_des_range=(args.z_des_lo, args.z_des_hi),
            vx_des_range=(args.vx_des_lo, args.vx_des_hi),
            randomize_task=True,
        )
        env = HopperEnv(cfg)
        env = Monitor(env)
        env.reset(seed=args.seed + rank)
        return env
    return _init


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = os.path.join(args.outdir, args.run_name)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    tb_dir = os.path.join(outdir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Vectorised env (parallel rollouts in a single process). DummyVecEnv is
    # simpler than SubprocVecEnv and our env step is cheap enough that the
    # GIL isn't the bottleneck here.
    venv = DummyVecEnv([make_env_factory(args, i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    policy_kwargs = dict(
        net_arch=dict(pi=[args.hidden_dim, args.hidden_dim],
                      vf=[args.hidden_dim, args.hidden_dim]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=ckpt_dir, name_prefix="ppo",
    )

    print(f"Training PPO  run={args.run_name}  n_envs={args.n_envs}  "
          f"total_timesteps={args.total_timesteps}")
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                progress_bar=False)

    model_path = os.path.join(outdir, "model.zip")
    norm_path = os.path.join(outdir, "vec_normalize.pkl")
    model.save(model_path)
    venv.save(norm_path)
    print(f"✓ model → {model_path}")
    print(f"✓ norm  → {norm_path}")


if __name__ == "__main__":
    main()
