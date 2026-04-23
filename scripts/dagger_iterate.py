"""
One iteration of DAgger:

  1. Roll out the current IL policy in MuJoCo (closed-loop).
  2. At each step, record the state and the EXPERT's action at that state
     (not the policy's — we want the correction).
  3. Save the resulting (state, expert_action) pairs, optionally merging
     with an existing demo file.

Usage:
    python scripts/dagger_iterate.py \\
        --policy-weights experiments/weights/il_energy_ftxty_v1.pt \\
        --policy-norm    experiments/weights/il_energy_ftxty_v1_norm.npz \\
        --policy ftxty \\
        --merge-with     data/il_demos_energy_combined.npz \\
        --out            data/il_demos_dagger_v1.npz \\
        --n-rollouts     80
"""

import argparse
import os
import time

import numpy as np
import torch
import mujoco

from hopper.il_policy import (
    ILPolicy, ILPolicyFTxTy, extract_il_state,
    IL_STATE_DIM, IL_ACTION_DIM,
)
from hopper.sim_data_collection import (
    PDGains, EnergyHopGains, HoppingState,
    energy_hopping_control,
    randomize_initial_state,
)
from hopper.simulation import actuator_name_to_id, get_body_pos_eul


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--policy-weights", required=True)
    p.add_argument("--policy-norm", required=True)
    p.add_argument("--policy", default="ftxty", choices=["wings", "ftxty"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--out", required=True,
                   help="Where to write the merged .npz")
    p.add_argument("--merge-with", default=None,
                   help="Optional existing .npz to concatenate with")
    p.add_argument("--n-rollouts", type=int, default=80)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--fps-ctrl", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--z-des-lo", type=float, default=0.05)
    p.add_argument("--z-des-hi", type=float, default=0.12)
    # Randomization: small enough that the EXPERT survives (so its labels are
    # trustworthy), while the POLICY may still fail — those failures are
    # exactly the corrections we want.
    p.add_argument("--tilt-deg", type=float, default=1.0)
    p.add_argument("--xy-range", type=float, default=0.005)
    p.add_argument("--z-init-lo", type=float, default=0.030)
    p.add_argument("--z-init-hi", type=float, default=0.050)
    p.add_argument("--vx-max", type=float, default=0.0,
                   help="Forward setpoint velocity range (m/s); per rollout "
                        "vx_des is sampled from [0, vx-max].")
    p.add_argument("--vx-bias-zero", type=float, default=0.25,
                   help="Fraction of rollouts forced to vx_des=0.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Load policy
    norm = np.load(args.policy_norm)
    X_mean = norm["X_mean"].astype(np.float32)
    X_std = norm["X_std"].astype(np.float32)
    if args.policy == "ftxty":
        policy = ILPolicyFTxTy(input_dim=IL_STATE_DIM,
                               hidden_dim=args.hidden_dim).to(args.device)
    else:
        policy = ILPolicy(input_dim=IL_STATE_DIM, output_dim=IL_ACTION_DIM,
                          hidden_dim=args.hidden_dim).to(args.device)
    policy.load(args.policy_weights, device=args.device)
    policy.eval()

    # MuJoCo
    mjmodel = mujoco.MjModel.from_xml_path(args.xml)
    mjdata = mujoco.MjData(mjmodel)
    mass = float(np.sum(mjmodel.body_mass))
    g = float(-mjmodel.opt.gravity[2])
    aids = [actuator_name_to_id(mjmodel, n) for n in ("f1", "f2", "f3", "f4")]
    dt_ctrl = 1.0 / args.fps_ctrl
    n_steps = int(args.duration * args.fps_ctrl)
    sub_steps = max(1, int(round(dt_ctrl / mjmodel.opt.timestep)))

    gains = PDGains()
    energy_gains = EnergyHopGains()

    X_all, Y_all = [], []
    # Per-rollout health stats
    tipped = 0
    n_full = 0

    started = time.time()
    for i in range(args.n_rollouts):
        randomize_initial_state(
            mjmodel, mjdata, rng,
            z_range=(args.z_init_lo, args.z_init_hi),
            tilt_deg=args.tilt_deg,
            xy_range=args.xy_range,
        )
        z_target = float(rng.uniform(args.z_des_lo, args.z_des_hi))
        # Sample forward velocity per rollout (matches collect_il_demos.py).
        if args.vx_max <= 0.0 or rng.random() < args.vx_bias_zero:
            vx_des = 0.0
        else:
            vx_des = float(rng.uniform(0.0, args.vx_max))
        # The expert now uses apex-feedback state; one HoppingState per rollout.
        hop_state = HoppingState(z_target)

        X_r = np.zeros((n_steps, IL_STATE_DIM), dtype=np.float32)
        Y_r = np.zeros((n_steps, IL_ACTION_DIM), dtype=np.float32)
        survived = n_steps
        for t_i in range(n_steps):
            t_sec = t_i * dt_ctrl
            x_des = vx_des * t_sec
            y_des = 0.0
            u_expert = energy_hopping_control(
                mjmodel, mjdata, z_target,
                mass, g, 0.015, 0.003, gains, energy_gains,
                state=hop_state,
                x_des=x_des, y_des=y_des,
            )
            state_vec = extract_il_state(
                mjmodel, mjdata, z_target,
                x_des=x_des, y_des=y_des,
            )
            X_r[t_i] = state_vec
            Y_r[t_i] = u_expert

            # Apply the POLICY's action (so the sim visits policy-induced states)
            s_n = (state_vec - X_mean) / X_std
            with torch.no_grad():
                x_t = torch.tensor(s_n, device=args.device).unsqueeze(0)
                if args.policy == "ftxty":
                    u_apply = policy.wing_forces(x_t).cpu().numpy()[0]
                else:
                    u_apply = policy(x_t).cpu().numpy()[0]

            for j, aid in enumerate(aids):
                mjdata.ctrl[aid] = float(u_apply[j])
            for _ in range(sub_steps):
                mujoco.mj_step(mjmodel, mjdata)

            pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
            if (abs(eul[0]) > np.deg2rad(60)
                    or abs(eul[1]) > np.deg2rad(60)
                    or abs(pos[0]) > 0.5 or abs(pos[1]) > 0.5):
                survived = t_i + 1
                tipped += 1
                break

        if survived == n_steps:
            n_full += 1
        X_all.append(X_r[:survived])
        Y_all.append(Y_r[:survived])

        print(f"[{i+1:3d}/{args.n_rollouts}] z_des={z_target*100:5.2f}cm  "
              f"survived={survived:4d}/{n_steps}  "
              f"{'full' if survived == n_steps else 'tipped'}")

    X_new = np.concatenate(X_all, axis=0).astype(np.float32)
    Y_new = np.concatenate(Y_all, axis=0).astype(np.float32)

    # Merge with previous demos if requested
    if args.merge_with is not None:
        old = np.load(args.merge_with)
        X = np.concatenate([old["X"].astype(np.float32), X_new], axis=0)
        Y = np.concatenate([old["Y"].astype(np.float32), Y_new], axis=0)
        merge_info = f"merged with {args.merge_with} ({len(old['X'])} samples)"
    else:
        X = X_new
        Y = Y_new
        merge_info = "no merge"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, X=X, Y=Y)

    elapsed = time.time() - started
    print(f"\nDAgger iteration complete in {elapsed:.1f}s")
    print(f"  rollouts: {args.n_rollouts} total, {n_full} full-length, {tipped} tipped")
    print(f"  new samples: {len(X_new)}")
    print(f"  {merge_info}")
    print(f"  total → {args.out}  X {X.shape}  Y {Y.shape}")


if __name__ == "__main__":
    main()
