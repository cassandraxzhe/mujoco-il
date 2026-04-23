"""
Closed-loop simulation with the IL policy (state -> action).
No CEM, no dynamics model — the policy is consulted directly each step.
"""

import argparse
import os

import numpy as np
import torch
import imageio
import mujoco

from hopper.il_policy import (
    ILPolicy, ILPolicyFTxTy, extract_il_state,
    IL_STATE_DIM, IL_ACTION_DIM,
)
from hopper.simulation import (
    body_name_to_id, actuator_name_to_id, get_body_pos_eul, get_system_com,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="assets/hopper.xml")
    p.add_argument("--run-name", default="il_v1")
    p.add_argument("--weights-dir", default="experiments/weights")
    p.add_argument("--video-dir", default="experiments/videos")
    p.add_argument("--sim-time", type=float, default=3.0)
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--z-des", type=float, default=0.08,
                   help="Hover setpoint (only used when --profile=hover)")
    p.add_argument("--profile", default="hover", choices=["hover", "step"],
                   help="z_des profile: constant hover or square-wave hopping")
    p.add_argument("--z-low", type=float, default=0.03,
                   help="Low setpoint for step profile (m)")
    p.add_argument("--z-high", type=float, default=0.10,
                   help="High setpoint for step profile (m)")
    p.add_argument("--dwell", type=float, default=0.5,
                   help="Seconds per half-period of step profile")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--policy", default="wings", choices=["wings", "ftxty"],
                   help="Must match what the weights were trained for.")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--x0", type=float, default=0.0,
                   help="Initial torso x position (m).")
    p.add_argument("--y0", type=float, default=0.0,
                   help="Initial torso y position (m).")
    p.add_argument("--z0", type=float, default=None,
                   help="Initial torso z position (m); default uses XML home pose.")
    p.add_argument("--vx-des", type=float, default=0.0,
                   help="Forward setpoint velocity (m/s). x_des(t)=vx_des·t. "
                        "Set to drive the policy toward a moving waypoint.")
    return p.parse_args()


def make_z_des_fn(args):
    if args.profile == "hover":
        return lambda t: args.z_des
    # step profile: alternate between z_low and z_high every `dwell` seconds
    def _z(t):
        phase = int(t // args.dwell) % 2
        return args.z_high if phase == 0 else args.z_low
    return _z


def main():
    args = parse_args()
    os.makedirs(args.video_dir, exist_ok=True)

    weights_path = os.path.join(args.weights_dir, f"{args.run_name}.pt")
    norm_path = os.path.join(args.weights_dir, f"{args.run_name}_norm.npz")

    norm = np.load(norm_path)
    X_mean = norm["X_mean"].astype(np.float32)
    X_std = norm["X_std"].astype(np.float32)

    if args.policy == "ftxty":
        policy = ILPolicyFTxTy(input_dim=IL_STATE_DIM,
                               hidden_dim=args.hidden_dim).to(args.device)
    else:
        policy = ILPolicy(input_dim=IL_STATE_DIM, output_dim=IL_ACTION_DIM,
                          hidden_dim=args.hidden_dim).to(args.device)
    policy.load(weights_path, device=args.device)
    policy.eval()

    mjmodel = mujoco.MjModel.from_xml_path(args.xml)
    mjdata = mujoco.MjData(mjmodel)
    # Optional initial-state override (for stair-env evaluation).
    if args.x0 != 0.0 or args.y0 != 0.0 or args.z0 is not None:
        mjdata.qpos[0] = args.x0
        mjdata.qpos[1] = args.y0
        if args.z0 is not None:
            mjdata.qpos[2] = args.z0
        mjdata.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(mjmodel, mjdata)
    renderer = mujoco.Renderer(mjmodel, width=640, height=480)

    actuator_ids = [actuator_name_to_id(mjmodel, n) for n in ("f1", "f2", "f3", "f4")]
    steps_per_frame = max(1, int(round((1.0 / args.fps) / mjmodel.opt.timestep)))
    n_frames = int(args.sim_time * args.fps)

    z_des_fn = make_z_des_fn(args)

    frames, zs, xs, ys, us, z_des_log = [], [], [], [], [], []
    for i in range(n_frames):
        t = i / args.fps
        z_des_t = z_des_fn(t)
        x_des_t = args.vx_des * t
        y_des_t = 0.0

        s = extract_il_state(mjmodel, mjdata, z_des_t,
                             x_des=x_des_t, y_des=y_des_t)
        s_n = (s - X_mean) / X_std
        with torch.no_grad():
            x_t = torch.tensor(s_n, device=args.device).unsqueeze(0)
            if args.policy == "ftxty":
                u = policy.wing_forces(x_t).cpu().numpy()[0]
            else:
                u = policy(x_t).cpu().numpy()[0]
        for j, aid in enumerate(actuator_ids):
            mjdata.ctrl[aid] = float(u[j])
        for _ in range(steps_per_frame):
            mujoco.mj_step(mjmodel, mjdata)

        bid = body_name_to_id(mjmodel, "hopper")
        p = mjdata.xpos[bid]
        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        us.append(u.copy()); z_des_log.append(z_des_t)

        if not args.no_video:
            renderer.update_scene(mjdata)
            frames.append(renderer.render())

        if i % 20 == 0 or i == n_frames - 1:
            print(f"Frame {i:3d}/{n_frames} | t={t:5.2f}s | "
                  f"z={p[2]:6.3f}m | z_des={z_des_t:5.3f}m | "
                  f"u={sum(u)*1000:6.2f}mN")

    zs = np.array(zs); xs = np.array(xs); ys = np.array(ys)
    U = np.array(us); Zdes = np.array(z_des_log)
    rmse = float(np.sqrt(np.mean((zs - Zdes) ** 2)))
    print(f"\nz  mean={zs.mean()*100:.2f} cm  std={zs.std()*100:.2f} cm  "
          f"RMSE (vs time-varying z_des)={rmse*100:.2f} cm")
    print(f"drift  x={xs[-1]*100:.2f} cm  y={ys[-1]*100:.2f} cm  "
          f"total={np.hypot(xs[-1], ys[-1])*100:.2f} cm")
    print(f"thrust  mean={U.mean()*1e3:.3f} mN/wing  max={U.max()*1e3:.3f} mN/wing")

    if args.profile == "step":
        # Phase-averaged tracking: mean z during each setpoint regime
        high_mask = Zdes > 0.5 * (args.z_low + args.z_high)
        if high_mask.any() and (~high_mask).any():
            print(f"phase z_high={args.z_high*100:.1f}cm: mean z="
                  f"{zs[high_mask].mean()*100:.2f}cm  std="
                  f"{zs[high_mask].std()*100:.2f}cm")
            print(f"phase z_low ={args.z_low*100:.1f}cm: mean z="
                  f"{zs[~high_mask].mean()*100:.2f}cm  std="
                  f"{zs[~high_mask].std()*100:.2f}cm")

    if not args.no_video:
        tag = args.profile
        video_path = os.path.join(args.video_dir, f"{args.run_name}_{tag}.mp4")
        imageio.mimsave(video_path, frames, fps=args.fps)
        print(f"video → {video_path}")


if __name__ == "__main__":
    main()
