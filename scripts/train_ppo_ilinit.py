"""
Train PPO with an IL-initialised policy network (warm-start).

Extends `scripts/train_ppo.py` with a `--il-init` flag that loads an
existing ILPolicyFTxTy's MLP weights into PPO's policy_net before
`model.learn()`. Seed VecNormalize's observation stats with the IL
training-time normalisation so the initial forward pass matches IL's
exactly, after which PPO's obs stats drift naturally under gradient
descent.

What's transferred:
  IL's Sequential[Linear, Tanh, Linear, Tanh, Linear]  →
    PPO's policy.mlp_extractor.policy_net[Linear, Tanh, Linear, Tanh]
  IL's input normalisation (X_mean, X_std)  →
    VecNormalize.obs_rms.mean / var

What's NOT transferred:
  IL's final Linear (3 outputs with sigmoid/tanh activation)  — left
  random because PPO's output is a Gaussian mean (unbounded) over
  actions in [−1, 1], which have different semantics from IL's raw
  pre-activation logits.
  IL's value head doesn't exist; PPO's value_net is left random.
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hopper.rl_env import HopperEnv, HopperEnvConfig
from hopper.il_policy import ILPolicyFTxTy, IL_STATE_DIM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--xml-mix", default=None)
    p.add_argument("--run-name", default="ppo_mix_ilinit_v1")
    p.add_argument("--outdir", default="experiments/ppo")
    p.add_argument("--il-weights", required=True,
                   help="Path to ILPolicyFTxTy weights (.pt). Associated "
                        "_norm.npz must be alongside.")
    p.add_argument("--il-hidden-dim", type=int, default=128)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--device", default="cpu")
    # Env knobs (forwarded to HopperEnvConfig)
    p.add_argument("--z-des-lo", type=float, default=0.06)
    p.add_argument("--z-des-hi", type=float, default=0.10)
    p.add_argument("--vx-des-lo", type=float, default=0.01)
    p.add_argument("--vx-des-hi", type=float, default=0.05)
    p.add_argument("--episode-sec", type=float, default=5.0)
    return p.parse_args()


def make_env_factory(args, rank):
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


def transfer_il_to_ppo(model, il_weights_path, il_hidden_dim):
    """Copy IL's MLP [Linear-Tanh-Linear-Tanh] into PPO's mlp_extractor.policy_net."""
    il_policy = ILPolicyFTxTy(input_dim=IL_STATE_DIM, hidden_dim=il_hidden_dim)
    il_policy.load(il_weights_path, device="cpu")

    il_net = il_policy.net       # Sequential[L, Tanh, L, Tanh, L]
    ppo_pi = model.policy.mlp_extractor.policy_net   # Sequential[L, Tanh, L, Tanh]

    # Shape check
    def linear_shape(layer): return (layer.out_features, layer.in_features)
    expected = [linear_shape(il_net[0]), linear_shape(il_net[2])]
    actual   = [linear_shape(ppo_pi[0]), linear_shape(ppo_pi[2])]
    if expected != actual:
        raise ValueError(
            f"Shape mismatch: IL layers {expected} vs PPO policy_net {actual}. "
            f"Re-train IL at the hidden_dim matching PPO's net_arch."
        )

    # Copy the two feature-extractor Linear layers
    with torch.no_grad():
        ppo_pi[0].weight.copy_(il_net[0].weight)
        ppo_pi[0].bias.copy_(il_net[0].bias)
        ppo_pi[2].weight.copy_(il_net[2].weight)
        ppo_pi[2].bias.copy_(il_net[2].bias)
    print(f"  ✓ Copied IL MLP layers into PPO policy_net "
          f"({expected[0][1]} → {expected[0][0]} → {expected[1][0]})")


def seed_obs_normalization(venv, norm_npz_path):
    """Overwrite VecNormalize's obs_rms with IL's training-time stats."""
    norm = np.load(norm_npz_path)
    rms = RunningMeanStd(shape=(IL_STATE_DIM,))
    rms.mean = norm["X_mean"].astype(np.float64)
    rms.var  = (norm["X_std"] ** 2).astype(np.float64)
    rms.count = 1_000.0   # pretend we've already seen data so updates don't swamp IL's stats
    venv.obs_rms = rms
    print(f"  ✓ Seeded VecNormalize obs_rms from {norm_npz_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = os.path.join(args.outdir, args.run_name)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    tb_dir = os.path.join(outdir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    venv = DummyVecEnv([make_env_factory(args, i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    # Seed VecNormalize's obs stats from IL's normalisation
    il_norm = args.il_weights.replace(".pt", "_norm.npz")
    if not os.path.exists(il_norm):
        raise FileNotFoundError(f"Expected IL norm stats at {il_norm}")
    seed_obs_normalization(venv, il_norm)

    policy_kwargs = dict(
        net_arch=dict(pi=[args.il_hidden_dim, args.il_hidden_dim],
                      vf=[args.il_hidden_dim, args.il_hidden_dim]),
        activation_fn=torch.nn.Tanh,
    )
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=10,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir, verbose=1,
        device=args.device, seed=args.seed,
    )

    # IL warm-start the policy MLP
    transfer_il_to_ppo(model, args.il_weights, args.il_hidden_dim)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=ckpt_dir, name_prefix="ppo",
    )

    print(f"\nTraining PPO (IL-init from {args.il_weights})  "
          f"run={args.run_name}  n_envs={args.n_envs}  "
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
