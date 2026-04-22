# Hopper Imitation Learning

Model Predictive Control (MPC) for a micro-scale flying insect-inspired hopper robot using imitation learning and learned dynamics.

## 🦗 Overview

This project implements an end-to-end pipeline for controlling a micro-robotic hopper:

1. **Data Collection**: Load experimental jumping data from real hardware
2. **Learning**: Train a neural network to learn forward dynamics
3. **Control**: Use Model Predictive Control (MPC) with Cross-Entropy Method (CEM) optimization
4. **Simulation**: Closed-loop control in MuJoCo with visualization

### The Hopper Robot

- **Mass**: ~1 mg (0.95 mg body + 0.20 mg leg)
- **Size**: 50mm × 50mm × 20mm
- **Actuators**: 4 wing thrusts + 1 spring leg
- **Control**: 80 Hz

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install numpy torch scipy mujoco matplotlib imageio

# Clone/navigate to project
cd mujoco-il
```

### Run Simulation

```python
from hopper import run_simulation

# Run closed-loop MPC simulation
results = run_simulation(
    hopper_xml="flying_hopper.xml",
    model_weights="hopper_mlp.pt",
    sim_time=5.0,
    z_des=0.08,  # Target height: 8 cm
    output_video="hopper_sim.mp4",
    plot=True
)
```

### Train Model

```python
from hopper import load_jumping_data, HopperMLP
import torch

# Load data
data = load_jumping_data("/path/to/jumping_data")
X, Y = data.X, data.Y

# Create and train model
model = HopperMLP(input_dim=14, output_dim=6, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(50):
    # ... training code ...
    pass

# Save model
torch.save(model.state_dict(), "hopper_mlp.pt")
```

## 📁 Project Structure

```
mujoco-il/
├── hopper/                    # Core library
│   ├── data_utils.py         # Data loading & preprocessing
│   ├── models.py             # Neural network (HopperMLP)
│   ├── mpc.py                # MPC & CEM optimization
│   ├── simulation.py         # MuJoCo simulation
│   ├── frame_alignment.py    # Frame transformations
│   └── evaluation.py         # Plotting & metrics
│
├── configs/
│   └── mpc_config.py         # MPC hyperparameters & presets
│
├── hopper-il.ipynb           # Original research notebook
├── hopper.xml                # MuJoCo model files
├── flying_hopper.xml
└── README.md                 # This file
```

## 🎛️ Configuration Presets

Three pre-configured modes for different behaviors:

### 1. Hopping Mode (Recommended)
```python
from configs.mpc_config import HoppingConfig

cfg = HoppingConfig()
# z_des = 0.08m (8 cm)
# Balanced weights for sustained hopping
```

### 2. High Jump Mode
```python
from configs.mpc_config import HighJumpConfig

cfg = HighJumpConfig()
# z_des = 0.20m (20 cm)
# Higher force limits and tracking weight
```

### 3. Conservative Mode
```python
from configs.mpc_config import ConservativeConfig

cfg = ConservativeConfig()
# z_des = 0.05m (5 cm)
# Maximum stability, minimal drift
```

## 🔧 Key Components

### 1. Data Loading

```python
from hopper import load_jumping_data

data = load_jumping_data(
    data_dir="/path/to/data",
    downsample_factor=5,      # 500 Hz → 100 Hz
    clip_start_sec=3.0,       # Remove initial transient
    clip_end_sec=13.0         # Remove final transient
)

# Access preprocessed data
X = data.X  # [N, 14]: [pos(3), eul(3), thrust(1), torque(3), signals(4)]
Y = data.Y  # [N, 6]: [delta_pos(3), delta_eul(3)]
```

### 2. Neural Network

```python
from hopper import HopperMLP

model = HopperMLP(
    input_dim=14,    # State + action
    output_dim=6,    # State change (delta)
    hidden_dim=32    # Hidden layer size
)

# Forward dynamics: delta = f(state, action)
delta = model(x)
next_state = current_state + delta
```

### 3. MPC with CEM

```python
from hopper import cem_optimize

# Optimize control sequence
u_optimal = cem_optimize(
    model,                 # Learned dynamics
    pos,                   # Current position [3]
    eul,                   # Current orientation [3]
    z_des=0.08,           # Desired height
    h=20,                  # Horizon (20 steps @ 80 Hz)
    pop=256,               # CEM population
    elites=32,             # CEM elites
    iters=4                # CEM iterations
)
```

### 4. Cost Function

The MPC cost function balances:

- **Attitude stability**: Minimize roll/pitch (stay upright)
- **Height tracking**: Track desired height z_des
- **Lateral drift**: Minimize x/y deviation
- **Control effort**: Minimize force usage
- **Angular velocity**: Dampen rotation

```python
from hopper import mpc_cost

cost = mpc_cost(
    traj_pos,     # Position trajectory [H, 3]
    traj_eul,     # Euler trajectory [H, 3]
    traj_u,       # Control trajectory [H, 4]
    z_des=0.08,   # Target height
    w_up=6.0,     # Attitude weight
    w_z=5.0,      # Height weight
    w_xy=2.0,     # Lateral weight
    w_u=2e-4,     # Control effort weight
    w_omega=2.0   # Angular velocity weight
)
```

## 🐛 Known Issues & Fixes

### Flying Hopper Bug

**Problem**: Hopper flies upward and off-screen instead of sustained hopping.

**Root Causes**:
1. `z_des = 0.20m` too high (20 cm)
2. Excessive hover bias added to CEM outputs
3. Force scale mismatch (CEM range 141× smaller than actual forces)

**Solution**:
```python
# Use HoppingConfig preset
from configs.mpc_config import HoppingConfig

cfg = HoppingConfig()
results = run_simulation(
    ...,
    z_des=cfg.Z_DES  # 0.08m instead of 0.20m
)
```

See [HOPPER_FIX_INSTRUCTIONS.md](HOPPER_FIX_INSTRUCTIONS.md) for details.

## 📊 Evaluation & Plotting

```python
from hopper import (
    evaluate_rollout,
    plot_training_history,
    plot_rollout_predictions,
    print_dataset_summary
)

# Dataset statistics
print_dataset_summary(X, Y)

# Training curves
plot_training_history(train_losses, val_losses)

# Model rollout evaluation
rollout_results = evaluate_rollout(model, X_test, Y_test, horizon=50)
plot_rollout_predictions(rollout_results)

# Control analysis
from hopper.evaluation import plot_control_forces
plot_control_forces(results['controls'])
```

## 🔬 Advanced: Frame Realignment

For orientation-robust control:

```python
from hopper.frame_alignment import (
    build_R_match,
    body_to_nn,
    tau_nn_to_body,
    allocate_per_wing_forces
)

# Build alignment matrix
R_match = build_R_match(R_body, R_mpc_1)

# Transform to NN frame
q_nn, omega_nn = body_to_nn(R_body, omega_body, R_match)

# Transform torque back to body frame
tau_body = tau_nn_to_body(tau_nn, R_match)

# Allocate per-wing forces
f_wings = allocate_per_wing_forces(F_cmd, Tx_cmd, Ty_cmd)
```

## 📖 Documentation

- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)**: Migrate from old notebook to new modular code
- **[HOPPER_FIX_INSTRUCTIONS.md](HOPPER_FIX_INSTRUCTIONS.md)**: Fix the flying hopper bug
- **[hopper_mpc_fixed.py](hopper_mpc_fixed.py)**: Fixed MPC implementation with ground detection

## 🧪 Testing

```python
# Quick sanity check
from hopper import *
from configs.mpc_config import print_config

# Show configuration
print_config('hopping')

# Load model
model = HopperMLP(14, 6)
model.load_state_dict(torch.load("hopper_mlp.pt"))

# Test MPC step
u = cem_optimize(model, pos=[0,0,0.05], eul=[0,0,0])
print(f"Control output: {u}")
```

## 📚 References

This implements model-based imitation learning with MPC:

1. **Imitation Learning**: Learn dynamics from expert demonstrations (real hardware data)
2. **Model Predictive Control**: Optimize action sequences using learned model
3. **Cross-Entropy Method**: Derivative-free optimization for MPC

Key algorithms:
- Neural network: MLP with 2 hidden layers (32 units each)
- Dynamics: Forward model predicts state changes
- Optimization: CEM with population 256, elites 32, 4 iterations
- Cost: Multi-objective (attitude, height, drift, effort, velocity)

## 🤝 Contributing

The modular structure makes it easy to extend:

- **New cost functions**: Add to `hopper/mpc.py`
- **New controllers**: Add to `hopper/simulation.py`
- **New models**: Add to `hopper/models.py`
- **New presets**: Add to `configs/mpc_config.py`

---

**Status**: ✅ Refactored, Documented, Bug-Fixed

**Last Updated**: 2025-01-07
