"""
Collect simulated rollouts under a PD controller and save X/Y pairs.

Usage:
    python scripts/collect_sim_data.py --n-rollouts 50 --out data/sim_v1.npz

Writes a single .npz containing:
    X [N, 14], Y [N, 6], meta (dict)
"""

import argparse
import os
import time

import numpy as np
import mujoco

from hopper.sim_data_collection import (
    PDGains,
    make_hover_profile,
    make_step_profile,
    randomize_initial_state,
    run_rollout,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--out", default="data/sim_v1.npz")
    p.add_argument("--n-rollouts", type=int, default=50)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--fps-ctrl", type=float, default=100.0)
    p.add_argument("--action-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # Init / target ranges (control coverage of state space)
    p.add_argument("--z-init-lo", type=float, default=0.03)
    p.add_argument("--z-init-hi", type=float, default=0.10)
    p.add_argument("--tilt-deg", type=float, default=5.0)
    p.add_argument("--z-hover-lo", type=float, default=0.04)
    p.add_argument("--z-hover-hi", type=float, default=0.12)
    p.add_argument("--z-step-lo-lo", type=float, default=0.01)
    p.add_argument("--z-step-lo-hi", type=float, default=0.04)
    p.add_argument("--z-step-hi-lo", type=float, default=0.08)
    p.add_argument("--z-step-hi-hi", type=float, default=0.14)
    # PD gains
    p.add_argument("--kp-z", type=float, default=0.05)
    p.add_argument("--kd-z", type=float, default=0.02)
    p.add_argument("--kp-att", type=float, default=2e-4)
    p.add_argument("--kd-att", type=float, default=2e-5)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    gains = PDGains(Kp_z=args.kp_z, Kd_z=args.kd_z,
                    Kp_att=args.kp_att, Kd_att=args.kd_att)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml)
    mjdata = mujoco.MjData(mjmodel)

    X_all, Y_all = [], []
    n_healthy = 0
    n_partial = 0
    started = time.time()

    for i in range(args.n_rollouts):
        randomize_initial_state(
            mjmodel, mjdata, rng,
            z_range=(args.z_init_lo, args.z_init_hi),
            tilt_deg=args.tilt_deg,
        )

        # Half hover, half step-hopping
        if i % 2 == 0:
            z_target = rng.uniform(args.z_hover_lo, args.z_hover_hi)
            z_des_fn = make_hover_profile(z_target, rng=rng)
            profile = "hover"
        else:
            z_low = rng.uniform(args.z_step_lo_lo, args.z_step_lo_hi)
            z_high = rng.uniform(args.z_step_hi_lo, args.z_step_hi_hi)
            dwell = rng.uniform(0.3, 0.8)
            z_des_fn = make_step_profile(z_low, z_high, dwell=dwell, rng=rng)
            profile = "step"

        res = run_rollout(
            mjmodel, mjdata, z_des_fn,
            duration=args.duration,
            fps_ctrl=args.fps_ctrl,
            action_noise=args.action_noise,
            gains=gains,
            rng=rng,
        )

        flag = "OK" if res.healthy else f"PARTIAL ({len(res.X)} samples)"
        print(f"[{i+1:3d}/{args.n_rollouts}] {profile:5s}  "
              f"samples={len(res.X):4d}  healthy={res.healthy}  {flag}")

        if res.healthy:
            n_healthy += 1
        elif len(res.X) > 0:
            n_partial += 1

        # Keep even partial rollouts — they contain valid transition data up
        # to the point where the robot flipped.
        if len(res.X) > 0:
            X_all.append(res.X)
            Y_all.append(res.Y)

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    Y = np.concatenate(Y_all, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        X=X, Y=Y,
        fps_ctrl=args.fps_ctrl,
        n_rollouts=args.n_rollouts,
        n_healthy=n_healthy,
        n_partial=n_partial,
        action_noise=args.action_noise,
    )

    elapsed = time.time() - started
    print(f"\nSaved {len(X)} samples  |  healthy={n_healthy}  "
          f"partial={n_partial}  "
          f"X {X.shape}  Y {Y.shape}  "
          f"→ {args.out}   ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
