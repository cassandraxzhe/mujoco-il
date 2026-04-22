"""
Hopper Imitation Learning Library

A modular library for training and controlling a MuJoCo hopper using
imitation learning and model predictive control.
"""

__version__ = "0.1.0"

# Core modules
from .models import HopperMLP
from .data_utils import JumpingData, HopperDataset, load_jumping_data
from .mpc import cem_optimize, mpc_cost, mpc_step, predict_next, state_action_to_input
from .simulation import run_simulation, get_body_pos_eul, apply_ctrl_to_data
from .frame_alignment import (
    build_R_match, body_to_nn, tau_nn_to_body,
    allocate_per_wing_forces, get_body_rotation_matrix,
    set_wing_forces_by_name
)
from .evaluation import (
    evaluate_rollout, compute_metrics,
    plot_training_history, plot_rollout_predictions,
    plot_data_distribution, print_dataset_summary, print_model_summary
)

__all__ = [
    # Models
    "HopperMLP",

    # Data
    "JumpingData",
    "HopperDataset",
    "load_jumping_data",

    # MPC
    "cem_optimize",
    "mpc_cost",
    "mpc_step",
    "predict_next",
    "state_action_to_input",

    # Simulation
    "run_simulation",
    "get_body_pos_eul",
    "apply_ctrl_to_data",

    # Frame alignment
    "build_R_match",
    "body_to_nn",
    "tau_nn_to_body",
    "allocate_per_wing_forces",
    "get_body_rotation_matrix",
    "set_wing_forces_by_name",

    # Evaluation
    "evaluate_rollout",
    "compute_metrics",
    "plot_training_history",
    "plot_rollout_predictions",
    "plot_data_distribution",
    "print_dataset_summary",
    "print_model_summary",
]
