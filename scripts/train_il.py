"""
Train an imitation-learning policy (state -> per-wing forces) on PD demos.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hopper.il_policy import (
    ILPolicy, ILPolicyFTxTy, wing_forces_to_ftxty,
    IL_STATE_DIM, IL_ACTION_DIM,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="data/il_demos_v1.npz")
    p.add_argument("--run-name", default="il_v1")
    p.add_argument("--weights-dir", default="experiments/weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--policy", default="wings", choices=["wings", "ftxty"],
                   help="wings: MLP predicts 4 per-wing forces directly. "
                        "ftxty: MLP predicts (F, Tx, Ty) and analytical mixer "
                        "derives wing forces — enforces physical symmetry.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.weights_dir, exist_ok=True)
    weights_path = os.path.join(args.weights_dir, f"{args.run_name}.pt")
    norm_path = os.path.join(args.weights_dir, f"{args.run_name}_norm.npz")

    d = np.load(args.demos)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    assert X.shape[1] == IL_STATE_DIM
    assert Y.shape[1] == IL_ACTION_DIM
    print(f"Loaded {len(X)} demo samples  |  X {X.shape}  Y {Y.shape}")

    if args.policy == "ftxty":
        # Convert wing-force demos to (F, Tx, Ty) targets via the mixer inverse.
        Y = wing_forces_to_ftxty(Y).astype(np.float32)
        print(f"  → (F, Tx, Ty) target:  Y {Y.shape}   "
              f"F range [{Y[:,0].min()*1000:.2f}, {Y[:,0].max()*1000:.2f}] mN   "
              f"|Tx|<{np.abs(Y[:,1]).max()*1e6:.2f}e-6 N·m   "
              f"|Ty|<{np.abs(Y[:,2]).max()*1e6:.2f}e-6 N·m")

    # Demos are episodic and independent across rollouts, but within a rollout
    # consecutive samples correlate. We do a per-rollout-free random split
    # here — cheaper than tracking rollout ids and fine for a small MLP.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    X = X[perm]; Y = Y[perm]
    n_test = int(0.10 * len(X))
    n_val = int(0.10 * len(X))
    n_train = len(X) - n_test - n_val
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xva, Yva = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    Xte, Yte = X[n_train + n_val:], Y[n_train + n_val:]

    # Normalize inputs only (outputs are bounded in [0, fmax] by the policy)
    X_mean = Xtr.mean(axis=0); X_std = Xtr.std(axis=0) + 1e-8
    np.savez(norm_path, X_mean=X_mean, X_std=X_std)
    print(f"Normalization saved → {norm_path}")

    Xtr_n = (Xtr - X_mean) / X_std
    Xva_n = (Xva - X_mean) / X_std
    Xte_n = (Xte - X_mean) / X_std

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr_n), torch.tensor(Ytr)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(Xva_n), torch.tensor(Yva)),
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(Xte_n), torch.tensor(Yte)),
        batch_size=args.batch_size, shuffle=False,
    )

    if args.policy == "ftxty":
        model = ILPolicyFTxTy(input_dim=IL_STATE_DIM,
                              hidden_dim=args.hidden_dim).to(device)
        # Per-dim loss weights for (F, Tx, Ty) so the 100× scale gap between
        # thrust and torque doesn't let F dominate the MSE. Floor the per-dim
        # std by the expected physical scale of each dimension so the weight
        # stays bounded even when one channel is near-constant in the data
        # (e.g. Tx/Ty ≈ 0 when training rollouts start centered).
        # Physical scales: F up to 4·fmax=0.012, T up to 2·L·fmax=9e-5.
        y_std = Ytr.std(axis=0)
        scale_floors = np.array([0.012 * 0.1, 9e-5 * 0.1, 9e-5 * 0.1],
                                dtype=np.float32)
        y_std_eff = np.maximum(y_std, scale_floors)
        loss_w = torch.tensor(1.0 / (y_std_eff ** 2),
                              dtype=torch.float32, device=device)
        print(f"ftxty Y std (raw):    {y_std}")
        print(f"ftxty Y std (floored):{y_std_eff}")
        print(f"ftxty loss weights:   {loss_w.cpu().numpy()}")

        def weighted_mse(pred, target):
            return ((pred - target) ** 2 * loss_w).mean()
        crit = weighted_mse
    else:
        model = ILPolicy(input_dim=IL_STATE_DIM, output_dim=IL_ACTION_DIM,
                         hidden_dim=args.hidden_dim).to(device)
        crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr += loss.item() * xb.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va += crit(model(xb), yb).item() * xb.size(0)
        va /= len(val_loader.dataset)

        if va < best_val:
            best_val = va
            model.save(weights_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"train={tr:.4e}  val={va:.4e}")

    print(f"\n✓ Best val: {best_val:.4e}")
    print(f"✓ Weights → {weights_path}")

    model.load(weights_path, device=device)
    model.eval()
    te = 0.0
    max_err = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            te += crit(pred, yb).item() * xb.size(0)
            max_err = max(max_err, (pred - yb).abs().max().item())
    te /= len(test_loader.dataset)
    # mN-scale MAE for readability
    mae_mn = np.sqrt(te) * 1000  # rough: MSE^.5 in force units
    print(f"Test MSE: {te:.4e}   approx RMSE: {np.sqrt(te)*1000:.3f} mN   "
          f"max abs error: {max_err*1000:.3f} mN")


if __name__ == "__main__":
    main()
