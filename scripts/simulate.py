"""
CLI simulation script — run closed-loop MPC in MuJoCo with a trained model.

Usage examples:
    python scripts/simulate.py
    python scripts/simulate.py --run-name hopping_v2
    python scripts/simulate.py --config high_jump --sim-time 10.0
    python scripts/simulate.py --run-name hopping_v1 --no-video --no-plots
    python scripts/simulate.py --list-configs

Expects the following files to exist (produced by scripts/train.py):
    experiments/weights/<run_name>.pt
    experiments/weights/<run_name>_norm.npz
"""

import argparse
import os

import numpy as np
import torch

from hopper import HopperMLP, run_simulation
from hopper.evaluation import plot_control_forces
from hopper.mpc import set_normalization_stats
from configs.mpc_config import load_config, print_config


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run MuJoCo hopper simulation with a trained MPC model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run identification
    p.add_argument("--run-name", default="hopping_v1",
                   help="Stem of the weights / norm files in experiments/weights/")
    p.add_argument("--weights-dir", default="experiments/weights")

    # MuJoCo model
    p.add_argument("--xml", default="assets/hopper.xml",
                   help="Path to MuJoCo XML model file")

    # MPC config
    p.add_argument("--config", default="hopping",
                   choices=["default", "hopping", "high_jump", "conservative"],
                   help="MPC preset to use for z_des and cost weights")
    p.add_argument("--z-des", type=float, default=None,
                   help="Override z_des from config (meters)")

    # Simulation
    p.add_argument("--sim-time", type=float, default=5.0,
                   help="Simulation duration in seconds")
    p.add_argument("--fps", type=int, default=80,
                   help="Video frame rate (also sets control frequency)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="Torch device (MuJoCo is CPU-bound; GPU rarely helps)")

    # Output
    p.add_argument("--video-dir",   default="experiments/videos")
    p.add_argument("--figures-dir", default="experiments/figures")
    p.add_argument("--no-video",    action="store_true",
                   help="Skip video recording")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip all matplotlib plots")
    p.add_argument("--no-realign",  action="store_true",
                   help="Disable frame realignment (debug option)")

    # Convenience
    p.add_argument("--list-configs", action="store_true",
                   help="Print all available config presets and exit")
    p.add_argument("--compare-all", action="store_true",
                   help="Run all three presets and print a summary table")

    return p.parse_args()


# ============================================================================
# HELPERS
# ============================================================================

def load_run(run_name, weights_dir, device):
    """Load model weights and normalization stats for a given run name."""
    weights_path = os.path.join(weights_dir, f"{run_name}.pt")
    norm_path    = os.path.join(weights_dir, f"{run_name}_norm.npz")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Run  python scripts/train.py --run-name {run_name}  first."
        )
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization stats not found: {norm_path}\n"
            f"Re-run training to regenerate them."
        )

    norm = np.load(norm_path)
    set_normalization_stats(
        norm["X_mean"], norm["X_std"],
        norm["Y_mean"], norm["Y_std"],
    )

    model = HopperMLP(input_dim=14, output_dim=6, hidden_dim=32)
    model.load(weights_path, device=device)
    model.eval()

    print(f"✓ Loaded weights:  {weights_path}")
    print(f"✓ Loaded norm:     {norm_path}")
    return model, weights_path


def print_result_summary(run_name, config_name, results, cfg):
    """Print a compact statistics block after a simulation run."""
    z   = np.array(results["torso_com"]["z"])
    x   = np.array(results["torso_com"]["x"])
    y   = np.array(results["torso_com"]["y"])
    U   = np.array(results["controls"])

    rmse = np.sqrt(np.mean((z - cfg["Z_DES"]) ** 2))

    print(f"\n{'='*52}")
    print(f"  Run: {run_name}  |  Config: {config_name}")
    print(f"{'='*52}")
    print(f"  Height   mean={z.mean()*100:.2f} cm  "
          f"std={z.std()*100:.2f} cm  "
          f"RMSE={rmse*100:.2f} cm")
    print(f"  Drift    x={x[-1]*100:.2f} cm  "
          f"y={y[-1]*100:.2f} cm  "
          f"total={np.hypot(x[-1], y[-1])*100:.2f} cm")
    print(f"  Force    mean={U.mean()*1e3:.3f} mN/wing  "
          f"max={U.max()*1e3:.3f} mN/wing")
    print(f"{'='*52}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # --list-configs shortcut
    if args.list_configs:
        for name in ["default", "hopping", "high_jump", "conservative"]:
            print_config(name)
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.video_dir,   exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # Validate XML
    if not os.path.exists(args.xml):
        raise FileNotFoundError(f"MuJoCo XML not found: {args.xml}")

    # ------------------------------------------------------------------
    # Single run (default)
    # ------------------------------------------------------------------
    if not args.compare_all:
        model, weights_path = load_run(args.run_name, args.weights_dir, args.device)

        cfg   = load_config(args.config)
        z_des = args.z_des if args.z_des is not None else cfg["Z_DES"]
        print_config(args.config)

        video_path = (
            None if args.no_video
            else os.path.join(args.video_dir,
                              f"{args.run_name}_{args.config}.mp4")
        )

        results = run_simulation(
            hopper_xml=args.xml,
            model_weights=weights_path,
            sim_time=args.sim_time,
            fps=args.fps,
            z_des=z_des,
            output_video=video_path,
            device=args.device,
            verbose=True,
            plot=False,
            use_frame_realignment=not args.no_realign,
        )

        print_result_summary(args.run_name, args.config, results, cfg)

        if video_path:
            print(f"✓ Video saved → {video_path}")

        if not args.no_plots:
            import matplotlib.pyplot as plt

            # Height tracking
            torso  = results["torso_com"]
            system = results["system_com"]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(torso["t"],  np.array(torso["z"])  * 100, label="Torso",  lw=2)
            ax.plot(system["t"], np.array(system["z"]) * 100, label="System", lw=2,
                    linestyle="--", alpha=0.7)
            ax.axhline(z_des * 100, color="r", linestyle=":", lw=2,
                       label=f"Target ({z_des*100:.0f} cm)")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Height [cm]")
            ax.set_title(f"Height Tracking — {args.run_name} / {args.config}")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(args.figures_dir,
                                    f"{args.run_name}_{args.config}_height.png")
            plt.savefig(fig_path, dpi=150)
            print(f"✓ Figure saved → {fig_path}")
            plt.show()

            # Control forces
            plot_control_forces(results["controls"], dt=1.0 / args.fps)
            fig_path = os.path.join(args.figures_dir,
                                    f"{args.run_name}_{args.config}_controls.png")
            plt.savefig(fig_path, dpi=150)
            print(f"✓ Figure saved → {fig_path}")
            plt.show()

    # ------------------------------------------------------------------
    # Compare all presets
    # ------------------------------------------------------------------
    else:
        model, weights_path = load_run(args.run_name, args.weights_dir, args.device)
        print(f"\nComparing all configs for run: {args.run_name}\n")

        rows = []
        for config_name in ["hopping", "high_jump", "conservative"]:
            cfg   = load_config(config_name)
            z_des = cfg["Z_DES"]

            video_path = (
                None if args.no_video
                else os.path.join(args.video_dir,
                                  f"{args.run_name}_{config_name}.mp4")
            )

            print(f"Running: {config_name}...")
            results = run_simulation(
                hopper_xml=args.xml,
                model_weights=weights_path,
                sim_time=args.sim_time,
                fps=args.fps,
                z_des=z_des,
                output_video=video_path,
                device=args.device,
                verbose=False,
                plot=False,
                use_frame_realignment=not args.no_realign,
            )

            z    = np.array(results["torso_com"]["z"])
            rmse = np.sqrt(np.mean((z - z_des) ** 2))
            rows.append((config_name, z_des, z.mean(), z.std(), rmse))

            if video_path:
                print(f"  ✓ Video → {video_path}")

        # Summary table
        print(f"\n{'Config':<14} {'z_des':>8} {'mean_z':>8} "
              f"{'std_z':>8} {'RMSE':>8}")
        print("-" * 52)
        for config_name, z_des, z_mean, z_std, rmse in rows:
            print(f"{config_name:<14} "
                  f"{z_des*100:>7.1f}cm "
                  f"{z_mean*100:>7.1f}cm "
                  f"{z_std*100:>7.1f}cm "
                  f"{rmse*100:>7.2f}cm")

        if not args.no_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 5))
            colors = ["steelblue", "darkorange", "seagreen"]
            for (config_name, z_des, *_), color in zip(rows, colors):
                # Re-use last results dict — rerun minimally for plot data
                cfg = load_config(config_name)
                res = run_simulation(
                    hopper_xml=args.xml,
                    model_weights=weights_path,
                    sim_time=args.sim_time,
                    fps=args.fps,
                    z_des=cfg["Z_DES"],
                    output_video=None,
                    device=args.device,
                    verbose=False,
                    plot=False,
                )
                t = res["torso_com"]["t"]
                z = np.array(res["torso_com"]["z"])
                ax.plot(t, z * 100, label=config_name, color=color, lw=2)
                ax.axhline(cfg["Z_DES"] * 100, color=color,
                           linestyle=":", alpha=0.5)

            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Height [cm]")
            ax.set_title(f"Config Comparison — {args.run_name}")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(args.figures_dir,
                                    f"{args.run_name}_comparison.png")
            plt.savefig(fig_path, dpi=150)
            print(f"\n✓ Comparison figure → {fig_path}")
            plt.show()


if __name__ == "__main__":
    main()