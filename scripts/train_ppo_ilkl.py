"""
PPO with BC (behaviour-cloning) regularisation toward a fixed IL policy.

This is a direct follow-up to the §43 negative result: naive IL weight
transfer converged to the same training reward as random init but
regressed on stair tasks (the IL-trained boldness was washed out by
PPO's gradient pressure). The fix this script implements is to:

    1. Initialise PPO's policy MLP from IL weights (same as §43).
    2. Add an auxiliary BC loss during every PPO update:
           L_BC = ||mean(π_PPO(s)) − a_IL(s)||²
       pulling PPO's policy mean toward the IL policy's action at
       every observation in the rollout buffer.
    3. Optionally decay the BC coefficient over training so PPO
       gradually gets more freedom to explore beyond IL.

The IL policy is frozen (no gradients) and evaluated with the same
observation-normalisation stats that PPO sees (both seeded from IL's
training-time normalisation, which VecNormalize then drifts from — a
mild approximation that's acceptable given VecNormalize.obs_rms drifts
only slowly on 1 M-step runs).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hopper.rl_env import HopperEnv, HopperEnvConfig
from hopper.il_policy import ILPolicyFTxTy, IL_STATE_DIM


# Environment action-space scaling: these have to match HopperEnv.step()
FMAX  = 0.003
L_ARM = 0.015
F_MAX = 4.0 * FMAX          # PPO action[0] maps to F ∈ [0, F_MAX]
T_MAX = 2.0 * L_ARM * FMAX  # PPO action[1,2] map to T ∈ ±T_MAX


def il_ftxty_to_action(il_ftxty):
    """Convert IL's (F, Tx, Ty) output (physical units) to PPO's
    [-1, 1] normalised action: inverse of HopperEnv's env.step() mapping.

    The env does:
        F  = (a[0] + 1) / 2 · F_MAX
        Tx = a[1] · T_MAX
        Ty = a[2] · T_MAX
    So:
        a[0] = 2·F/F_MAX − 1
        a[1] = Tx / T_MAX
        a[2] = Ty / T_MAX
    """
    return torch.stack([
        2.0 * il_ftxty[..., 0] / F_MAX - 1.0,
        il_ftxty[..., 1] / T_MAX,
        il_ftxty[..., 2] / T_MAX,
    ], dim=-1)


class PPOWithIL(PPO):
    """PPO with an auxiliary BC loss against a fixed IL policy.

    Overrides `train()` to add `bc_coef · ||π_mean − a_IL||²` per
    minibatch. Everything else — clipped surrogate loss, value loss,
    entropy bonus, gradient clipping — is unchanged from SB3's PPO.
    """

    def __init__(self, *args, il_policy, bc_coef=1.0, bc_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.il_policy = il_policy
        self.il_policy.eval()
        for p in self.il_policy.parameters():
            p.requires_grad = False
        self.bc_coef = float(bc_coef)
        self.bc_decay = float(bc_decay)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, bc_losses = [], [], [], []
        clip_fractions, approx_kl_divs_all = [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -entropy.mean()

                # --- BC loss against IL policy ---
                dist = self.policy.get_distribution(rollout_data.observations)
                # DiagGaussianDistribution wraps torch.distributions.Normal
                policy_mean = dist.distribution.mean
                with torch.no_grad():
                    il_ftxty = self.il_policy.forward(rollout_data.observations)
                    il_action = il_ftxty_to_action(il_ftxty).clamp(-1.0, 1.0)
                bc_loss = F.mse_loss(policy_mean, il_action)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.bc_coef * bc_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                bc_losses.append(bc_loss.item())
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

            self._n_updates += 1
            approx_kl_divs_all.extend(approx_kl_divs)

        # Decay BC coefficient for next train() call
        self.bc_coef *= self.bc_decay

        # Log custom metrics into the SB3 logger
        self.logger.record("train/bc_loss", float(np.mean(bc_losses)))
        self.logger.record("train/bc_coef", self.bc_coef)
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs_all)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def transfer_il_to_ppo(model, il_policy, il_hidden_dim):
    il_net = il_policy.net
    ppo_pi = model.policy.mlp_extractor.policy_net
    def lshape(l): return (l.out_features, l.in_features)
    if lshape(ppo_pi[0]) != lshape(il_net[0]) or lshape(ppo_pi[2]) != lshape(il_net[2]):
        raise ValueError(
            f"shape mismatch: IL {(lshape(il_net[0]), lshape(il_net[2]))} "
            f"vs PPO {(lshape(ppo_pi[0]), lshape(ppo_pi[2]))}"
        )
    with torch.no_grad():
        ppo_pi[0].weight.copy_(il_net[0].weight)
        ppo_pi[0].bias.copy_(il_net[0].bias)
        ppo_pi[2].weight.copy_(il_net[2].weight)
        ppo_pi[2].bias.copy_(il_net[2].bias)
    print(f"  ✓ Copied IL MLP into PPO policy_net")


def seed_obs_rms(venv, norm_npz_path):
    norm = np.load(norm_npz_path)
    rms = RunningMeanStd(shape=(IL_STATE_DIM,))
    rms.mean = norm["X_mean"].astype(np.float64)
    rms.var  = (norm["X_std"] ** 2).astype(np.float64)
    rms.count = 1_000.0
    venv.obs_rms = rms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--xml-mix", default=None)
    p.add_argument("--run-name", default="ppo_mix_ilkl_v1")
    p.add_argument("--outdir", default="experiments/ppo")
    p.add_argument("--il-weights", required=True)
    p.add_argument("--il-hidden-dim", type=int, default=128)
    p.add_argument("--bc-coef", type=float, default=1.0,
                   help="Initial BC-loss coefficient (weight on ||π_mean − a_IL||²).")
    p.add_argument("--bc-decay", type=float, default=1.0,
                   help="Multiplicative decay of bc-coef per PPO train() call. "
                        "1.0 = no decay, <1 = BC influence fades during training.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--z-des-lo", type=float, default=0.06)
    p.add_argument("--z-des-hi", type=float, default=0.10)
    p.add_argument("--vx-des-lo", type=float, default=0.01)
    p.add_argument("--vx-des-hi", type=float, default=0.05)
    p.add_argument("--episode-sec", type=float, default=5.0)
    return p.parse_args()


def make_env_factory(args, rank):
    xml = args.xml_mix if (args.xml_mix and rank % 2 == 1) else args.xml

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
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    outdir = os.path.join(args.outdir, args.run_name)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    tb_dir = os.path.join(outdir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(tb_dir, exist_ok=True)

    venv = DummyVecEnv([make_env_factory(args, i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    il_norm_path = args.il_weights.replace(".pt", "_norm.npz")
    if not os.path.exists(il_norm_path):
        raise FileNotFoundError(f"Need IL norm at {il_norm_path}")
    seed_obs_rms(venv, il_norm_path)

    # Load IL policy (frozen reference)
    il_policy = ILPolicyFTxTy(input_dim=IL_STATE_DIM, hidden_dim=args.il_hidden_dim)
    il_policy.load(args.il_weights, device=args.device)

    policy_kwargs = dict(
        net_arch=dict(pi=[args.il_hidden_dim, args.il_hidden_dim],
                      vf=[args.il_hidden_dim, args.il_hidden_dim]),
        activation_fn=torch.nn.Tanh,
    )
    model = PPOWithIL(
        "MlpPolicy", venv,
        il_policy=il_policy,
        bc_coef=args.bc_coef,
        bc_decay=args.bc_decay,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps, batch_size=args.batch_size,
        ent_coef=args.ent_coef, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, n_epochs=10,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir, verbose=1,
        device=args.device, seed=args.seed,
    )
    transfer_il_to_ppo(model, il_policy, args.il_hidden_dim)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=ckpt_dir, name_prefix="ppo",
    )

    print(f"\nTraining PPO+IL-BC  run={args.run_name}  bc_coef={args.bc_coef}  "
          f"bc_decay={args.bc_decay}  n_envs={args.n_envs}  "
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
