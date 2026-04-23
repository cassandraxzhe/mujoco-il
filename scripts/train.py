"""
CLI training script for the HopperMLP dynamics model.

Usage examples:
    python scripts/train.py
    python scripts/train.py --run-name hopping_v2 --epochs 150
    python scripts/train.py --config high_jump --lr 5e-4 --hidden-dim 64
    python scripts/train.py --data-dir /path/to/data --no-plots

Outputs (all under experiments/weights/):
    <run_name>.pt       — best model checkpoint (lowest val loss)
    <run_name>_norm.npz — normalization statistics for inference
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hopper import HopperDataset, HopperMLP, load_jumping_data
from hopper.evaluation import (
    plot_training_history,
    plot_rollout_predictions,
    evaluate_rollout,
    compute_metrics,
    print_model_summary,
    print_dataset_summary,
)
from hopper.mpc import set_normalization_stats
from configs.mpc_config import load_config


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train HopperMLP dynamics model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--data-dir", default="/Users/cassandrahe/MIT Dropbox/Cassandra He/jumping_data",
                   help="Directory containing .mat hardware data files")
    p.add_argument("--sim-data", default=None,
                   help="Optional .npz with additional X/Y from sim rollouts")
    p.add_argument("--run-name", default="hopping_v1",
                   help="Name for this run — used as filename stem for outputs")
    p.add_argument("--weights-dir", default="experiments/weights",
                   help="Directory to save model weights and norm stats")

    # Data preprocessing
    p.add_argument("--downsample", type=int, default=5,
                   help="Decimation factor (500 Hz → 100 Hz at factor 5)")
    p.add_argument("--clip-start", type=float, default=3.0,
                   help="Clip start time in seconds (removes initial transient)")
    p.add_argument("--clip-end", type=float, default=13.0,
                   help="Clip end time in seconds (removes final transient)")

    # Model
    p.add_argument("--hidden-dim", type=int, default=32,
                   help="Hidden layer width for HopperMLP")

    # Training
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate (Adam)")
    p.add_argument("--lr-patience", type=int, default=10,
                   help="ReduceLROnPlateau patience in epochs")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau reduction factor")
    p.add_argument("--seed", type=int, default=42)

    # Config (for display / reference only — not used during training itself)
    p.add_argument("--config", default="hopping",
                   choices=["default", "hopping", "high_jump", "conservative"],
                   help="MPC config preset to display alongside training")

    # Misc
    p.add_argument("--device", default=None,
                   help="torch device override (default: auto-detect cuda/cpu)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip all matplotlib plots (useful for headless runs)")
    p.add_argument("--rollout-horizon", type=int, default=50,
                   help="Horizon for post-training rollout evaluation")

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Device ---
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Output paths ---
    os.makedirs(args.weights_dir, exist_ok=True)
    weights_path = os.path.join(args.weights_dir, f"{args.run_name}.pt")
    norm_path    = os.path.join(args.weights_dir, f"{args.run_name}_norm.npz")
    print(f"Run:     {args.run_name}")
    print(f"Weights: {weights_path}")
    print(f"Norm:    {norm_path}\n")

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    print("Loading data...")
    all_data = load_jumping_data(
        data_dir=args.data_dir,
        downsample_factor=args.downsample,
        clip_start_sec=args.clip_start,
        clip_end_sec=args.clip_end,
    )

    X = all_data.X.astype(np.float32)
    Y = all_data.Y.astype(np.float32)
    print(f"Loaded {len(X)} hardware samples  |  X: {X.shape}  Y: {Y.shape}")

    X_sim = Y_sim = None
    if args.sim_data:
        sim = np.load(args.sim_data)
        X_sim = sim["X"].astype(np.float32)
        Y_sim = sim["Y"].astype(np.float32)
        if X_sim.shape[1] != X.shape[1] or Y_sim.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Sim shape mismatch: hardware X {X.shape} Y {Y.shape}, "
                f"sim X {X_sim.shape} Y {Y_sim.shape}"
            )
        print(f"Loaded {len(X_sim)} sim samples  →  will augment train split only "
              f"(val/test stay hardware-only for honest eval)")
    print()

    if not args.no_plots:
        print_dataset_summary(X, Y)

    # ------------------------------------------------------------------ #
    # 2. Temporal train / val / test split
    # ------------------------------------------------------------------ #
    n       = len(X)
    n_test  = int(0.20 * n)
    n_val   = int(0.16 * n)
    n_train = n - n_val - n_test

    X_train, Y_train = X[:n_train],              Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],        Y[n_train+n_val:]

    if X_sim is not None:
        # Sim augments train split only; shuffle sim so batches mix the two sources.
        X_train = np.concatenate([X_train, X_sim], axis=0)
        Y_train = np.concatenate([Y_train, Y_sim], axis=0)
        rng_mix = np.random.default_rng(args.seed)
        perm = rng_mix.permutation(len(X_train))
        X_train = X_train[perm]; Y_train = Y_train[perm]

    print(f"Split  →  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # ------------------------------------------------------------------ #
    # 3. Normalization (fit on train only)
    # ------------------------------------------------------------------ #
    X_mean = X_train.mean(axis=0);  X_std = X_train.std(axis=0) + 1e-8
    Y_mean = Y_train.mean(axis=0);  Y_std = Y_train.std(axis=0) + 1e-8

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std
    Y_train_n = (Y_train - Y_mean) / Y_std
    Y_val_n   = (Y_val   - Y_mean) / Y_std
    Y_test_n  = (Y_test  - Y_mean) / Y_std

    np.savez(norm_path, X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
    set_normalization_stats(X_mean, X_std, Y_mean, Y_std)
    print(f"Normalization stats saved → {norm_path}\n")

    # ------------------------------------------------------------------ #
    # 4. DataLoaders
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(HopperDataset(X_train_n, Y_train_n),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(HopperDataset(X_val_n,   Y_val_n),
                              batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(HopperDataset(X_test_n,  Y_test_n),
                              batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------ #
    # 5. Model, optimizer, scheduler
    # ------------------------------------------------------------------ #
    model = HopperMLP(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)

    print_model_summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=args.lr_patience,
        factor=args.lr_factor,
    )

    # ------------------------------------------------------------------ #
    # 6. Training loop
    # ------------------------------------------------------------------ #
    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f"Training for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_train = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), Y_b)
            loss.backward()
            optimizer.step()
            epoch_train += loss.item() * X_b.size(0)
        epoch_train /= len(train_loader.dataset)
        train_losses.append(epoch_train)

        # Validate
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                epoch_val += criterion(model(X_b), Y_b).item() * X_b.size(0)
        epoch_val /= len(val_loader.dataset)
        val_losses.append(epoch_val)

        scheduler.step(epoch_val)

        # Save best checkpoint
        if epoch_val < best_val:
            best_val = epoch_val
            model.save(weights_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={epoch_train:.6f}  val={epoch_val:.6f}")

    print(f"\n✓ Training complete  |  best val loss: {best_val:.6f}")
    print(f"✓ Weights saved → {weights_path}")

    # ------------------------------------------------------------------ #
    # 7. Evaluation
    # ------------------------------------------------------------------ #
    model.load(weights_path, device=device)

    test_metrics = compute_metrics(model, test_loader, device=device)
    print(f"\nTest set  |  MSE: {test_metrics['mse']:.6f}  "
          f"MAE: {test_metrics['mae']:.6f}")

    rollout = evaluate_rollout(
        model, X_test_n, Y_test_n,
        horizon=args.rollout_horizon,
        device=device,
    )
    print(f"Rollout MSE ({args.rollout_horizon} steps): {rollout['mse']:.6f}")

    # ------------------------------------------------------------------ #
    # 8. Plots (skipped with --no-plots)
    # ------------------------------------------------------------------ #
    if not args.no_plots:
        plot_training_history(train_losses, val_losses)
        state_labels = ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"]
        plot_rollout_predictions(rollout, state_labels=state_labels)


if __name__ == "__main__":
    main()