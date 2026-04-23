"""
Generate IL demonstrations by running the PD hover controller in MuJoCo
and logging (IL state, PD action) pairs.

Output: .npz with
    X [N, 7]  — IL state vector per timestep
    Y [N, 4]  — per-wing forces the PD commanded (the 'expert' action)
"""

import argparse
import os
import time

import numpy as np
import mujoco

from hopper.il_policy import extract_il_state, IL_STATE_DIM, IL_ACTION_DIM
from hopper.sim_data_collection import (
    PDGains, HopGains, HoppingState,
    EnergyHopGains,
    pd_control, hopping_control, energy_hopping_control,
    make_hover_profile, make_step_profile,
    randomize_initial_state,
)
from hopper.simulation import actuator_name_to_id, get_body_pos_eul


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--out", default="data/il_demos_v1.npz")
    p.add_argument("--n-rollouts", type=int, default=80)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--fps-ctrl", type=float, default=100.0)
    p.add_argument("--action-noise", type=float, default=0.0,
                   help="Optional noise added to the APPLIED action (sim only). "
                        "The logged action is the PD command (no noise).")
    p.add_argument("--seed", type=int, default=42)
    # z_des sweep
    p.add_argument("--z-des-lo", type=float, default=0.04)
    p.add_argument("--z-des-hi", type=float, default=0.12)
    p.add_argument("--tilt-deg", type=float, default=5.0)
    p.add_argument("--z-init-lo", type=float, default=0.03)
    p.add_argument("--z-init-hi", type=float, default=0.10)
    p.add_argument("--xy-range", type=float, default=0.05,
                   help="Initial x/y uniform half-range (m). 0 disables xy randomization.")
    # Expert selection
    p.add_argument("--expert", default="pd", choices=["pd", "hop", "energy"],
                   help="pd: cascaded hover PD. "
                        "hop: apex-regulated bang-bang hopping FSM. "
                        "energy: smooth energy-shaping hopping controller.")
    p.add_argument("--z-switch-frac", type=float, default=0.6)
    p.add_argument("--apex-band", type=float, default=0.005)
    p.add_argument("--K-E", type=float, default=300.0,
                   help="energy expert: F = K_E · (E_des − E)")
    p.add_argument("--F-min-frac", type=float, default=0.0,
                   help="energy expert: lower clip as fraction of 4·fmax")
    p.add_argument("--vx-max", type=float, default=0.0,
                   help="Per-rollout forward velocity sampled from "
                        "[0, vx-max] (m/s). x_des(t) = vx_des · t drives the "
                        "expert's lateral PD toward a moving target.")
    p.add_argument("--vx-bias-zero", type=float, default=0.25,
                   help="Fraction of rollouts forced to vx_des=0 (pure "
                        "hover). The rest sample vx_des ∈ (0, vx-max]. "
                        "Ensures the vx_des=0 regime is well-represented.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    gains = PDGains()
    hop_gains = HopGains(z_switch_frac=args.z_switch_frac,
                         apex_band=args.apex_band)
    energy_gains = EnergyHopGains(K_E=args.K_E, F_min_frac=args.F_min_frac)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml)
    mjdata = mujoco.MjData(mjmodel)
    actuator_ids = [actuator_name_to_id(mjmodel, n) for n in ("f1", "f2", "f3", "f4")]
    mass = float(np.sum(mjmodel.body_mass))
    g = float(-mjmodel.opt.gravity[2])
    L = 0.015
    fmax = 0.003

    dt_ctrl = 1.0 / args.fps_ctrl
    n_steps = int(args.duration * args.fps_ctrl)
    sub_steps = max(1, int(round(dt_ctrl / mjmodel.opt.timestep)))

    X_all, Y_all = [], []
    n_healthy = 0
    n_partial = 0
    started = time.time()

    for i in range(args.n_rollouts):
        randomize_initial_state(
            mjmodel, mjdata, rng,
            z_range=(args.z_init_lo, args.z_init_hi),
            tilt_deg=args.tilt_deg,
            xy_range=args.xy_range,
        )

        if args.expert in ("hop", "energy"):
            # Hopping experts use a single constant z_des per rollout.
            z_target = rng.uniform(args.z_des_lo, args.z_des_hi)
            z_des_fn = make_hover_profile(z_target, jitter_std=0.0, rng=rng)
            # Both experts need HoppingState now (energy expert uses apex feedback).
            hop_state = HoppingState(z_target)
            # Forward-velocity command: either pure-hover (vx_des=0) or a
            # random value in (0, vx_max]. Mixing both regimes in training
            # keeps the policy competent at pure hover while also learning
            # forward drive.
            if args.vx_max <= 0.0 or rng.random() < args.vx_bias_zero:
                vx_des = 0.0
            else:
                vx_des = rng.uniform(0.0, args.vx_max)
            profile = f"{args.expert}({z_target*100:.1f}cm, vx={vx_des*100:.1f}cm/s)"
        elif i % 2 == 0:
            z_target = rng.uniform(args.z_des_lo, args.z_des_hi)
            z_des_fn = make_hover_profile(z_target, rng=rng)
            hop_state = None
            profile = "hover"
        else:
            z_low = rng.uniform(0.02, 0.05)
            z_high = rng.uniform(0.08, args.z_des_hi)
            dwell = rng.uniform(0.3, 0.8)
            z_des_fn = make_step_profile(z_low, z_high, dwell=dwell, rng=rng)
            hop_state = None
            vx_des = 0.0
            profile = "step"

        X_r = np.zeros((n_steps, IL_STATE_DIM), dtype=np.float32)
        Y_r = np.zeros((n_steps, IL_ACTION_DIM), dtype=np.float32)
        healthy = True
        actual_n = 0
        for t_i in range(n_steps):
            t = t_i * dt_ctrl
            z_des = z_des_fn(t)
            x_des = vx_des * t          # forward setpoint advances linearly
            y_des = 0.0                 # no lateral commanded motion

            # Call the expert first so hop_state.last_apex reflects any apex
            # event detected at the current step (matches what simulate_il.py
            # does: apex-detection runs before the policy is queried).
            if args.expert == "hop":
                u_pd = hopping_control(mjmodel, mjdata, z_des, mass, g, L, fmax,
                                       gains, hop_gains, hop_state,
                                       x_des=x_des, y_des=y_des)
            elif args.expert == "energy":
                u_pd = energy_hopping_control(mjmodel, mjdata, z_des, mass, g,
                                              L, fmax, gains, energy_gains,
                                              state=hop_state,
                                              x_des=x_des, y_des=y_des)
            else:
                u_pd = pd_control(mjmodel, mjdata, z_des, mass, g, L, fmax,
                                  gains, x_des=x_des, y_des=y_des)

            # Log (state, expert action) with translation-invariant state.
            state_vec = extract_il_state(
                mjmodel, mjdata, z_des,
                x_des=x_des, y_des=y_des,
            )
            X_r[t_i] = state_vec
            Y_r[t_i] = u_pd

            # What we actually apply can have noise, to broaden visited states
            if args.action_noise > 0:
                u_apply = np.clip(
                    u_pd + rng.normal(0.0, args.action_noise * fmax, size=4),
                    0.0, fmax,
                ).astype(np.float32)
            else:
                u_apply = u_pd

            for j, aid in enumerate(actuator_ids):
                mjdata.ctrl[aid] = float(u_apply[j])
            for _ in range(sub_steps):
                mujoco.mj_step(mjmodel, mjdata)

            pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name="hopper")
            actual_n = t_i + 1
            if (abs(eul[0]) > np.deg2rad(60)
                    or abs(eul[1]) > np.deg2rad(60)
                    or abs(pos[0]) > 0.5 or abs(pos[1]) > 0.5):
                healthy = False
                break

        X_r = X_r[:actual_n]
        Y_r = Y_r[:actual_n]
        if len(X_r) > 0:
            X_all.append(X_r)
            Y_all.append(Y_r)

        if healthy:
            n_healthy += 1
        elif len(X_r) > 0:
            n_partial += 1

        print(f"[{i+1:3d}/{args.n_rollouts}] {profile:5s}  "
              f"samples={len(X_r):4d}  healthy={healthy}")

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    Y = np.concatenate(Y_all, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, X=X, Y=Y,
             fps_ctrl=args.fps_ctrl,
             n_rollouts=args.n_rollouts,
             n_healthy=n_healthy, n_partial=n_partial,
             action_noise=args.action_noise)

    elapsed = time.time() - started
    print(f"\nSaved {len(X)} samples  |  healthy={n_healthy}  "
          f"partial={n_partial}  X {X.shape}  Y {Y.shape}  "
          f"→ {args.out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
