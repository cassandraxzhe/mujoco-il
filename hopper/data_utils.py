"""
Data loading and preprocessing utilities for hopper experiments.
"""

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from scipy.signal import decimate
from pathlib import Path


class JumpingData:
    """
    Container for jumping hopper data from .mat files.

    Fields:
        .eul_ds: Downsampled Euler XYZ angles in radians
        .pos_ds: Downsampled position in meters
        .signals_ds: Downsampled control signals (4 wing commands)
        .torque: Torque data
        .thrust: Thrust data
        .time: Time array
        .X: Input features for ML [pos, eul, thrust, torque, signals]
        .Y: Output targets for ML [delta_pos, delta_eul]
    """

    def __init__(self, filename: str):
        self.name = filename
        self.dict = {}
        self.eul_ds, self.pos_ds, self.signals_ds = None, None, None
        self.torque, self.thrust, self.time = None, None, None
        self.X, self.Y = None, None
        self.clipped = False

    def load_from_dict(self, mat_dict: dict, downsample_factor: int = 5):
        """
        Load data from MATLAB dictionary structure.

        Args:
            mat_dict: Dictionary from scipy.io.loadmat
            downsample_factor: Decimation factor (default: 5)
        """
        self.dict = mat_dict

        # Extract raw arrays
        eul = self.dict["rst_Eul_XYZ"][0][0][1][0][0][0]  # (N,3)
        signals = self.dict["rst_driving_signals"][0][0][1][0][0][0]  # (N,4)
        torque = self.dict["rst_torque_b"][0][0][1][0][0][0]  # (N/5,4)
        pos = self.dict["rst_p_raw"][0][0][1][0][0][0]  # (N,3)
        thrust = np.transpose(self.dict["rst_thrust"][0][0][1][0][0][0][0])  # (N/5,1)

        # Downsample signals, euler, and position
        n_samples = signals.shape[0]
        signals_ds = decimate(signals, downsample_factor, axis=0, zero_phase=True)
        eul_ds = decimate(eul, downsample_factor, axis=0, zero_phase=True)
        pos_ds = decimate(pos, downsample_factor, axis=0, zero_phase=True)

        # Truncate to match torque/thrust length
        min_len = min(signals_ds.shape[0], torque.shape[0], thrust.shape[0])
        self.signals_ds = signals_ds[:min_len]
        self.eul_ds = eul_ds[:min_len]
        self.pos_ds = pos_ds[:min_len]
        self.torque = torque[:min_len]
        self.thrust = thrust[:min_len]

        # Create time array
        dt = 1.0 / 100.0  # 100 Hz after downsampling
        self.time = np.arange(min_len) * dt

    def clip(self, t_min: float, t_max: float, hz: int):
        """
        Clip data to specified time range (e.g., remove transients).

        Args:
            t_min: Start time in seconds
            t_max: End time in seconds
            hz: Sampling frequency in Hz

        """
        if not self.clipped:
            # Convert time to indices
            start_idx = int(t_min * hz)
            end_idx = int(t_max * hz)

            self.eul_ds = self.eul_ds[start_idx:end_idx]
            self.pos_ds = self.pos_ds[start_idx:end_idx]
            self.signals_ds = self.signals_ds[start_idx:end_idx]
            self.thrust = self.thrust[start_idx:end_idx]
            self.torque = self.torque[start_idx:end_idx]

            self.clipped = True

    def get_XY(self):
        """
        Construct input-output pairs for supervised learning.

        Returns:
            X: [N-1, 14] input features [pos, eul, thrust, torque, signals]
            Y: [N-1, 6] output targets [delta_pos, delta_eul]
        """
        N = self.pos_ds.shape[0]

        # Input: current state + action
        X = np.concatenate([
            self.pos_ds[:-1],      # [N-1, 3]
            self.eul_ds[:-1],      # [N-1, 3]
            self.thrust[:-1],      # [N-1, 1]
            self.torque[:-1, :3],  # [N-1, 3] (use first 3 torque components)
            self.signals_ds[:-1]   # [N-1, 4]
        ], axis=1)  # [N-1, 14]

        # Output: state change (delta)
        delta_pos = self.pos_ds[1:] - self.pos_ds[:-1]  # [N-1, 3]
        delta_eul = self.eul_ds[1:] - self.eul_ds[:-1]  # [N-1, 3]
        Y = np.concatenate([delta_pos, delta_eul], axis=1)  # [N-1, 6]

        self.X = X
        self.Y = Y
        return X, Y


class HopperDataset(Dataset):
    """
    PyTorch Dataset for hopper dynamics learning.

    Args:
        X: Input features [N, input_dim]
        Y: Output targets [N, output_dim]
    """

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_jumping_data(data_dir: str, downsample_factor: int = 5,
                      clip_start_sec: float = 3.0, clip_end_sec: float = 13.0):
    """
    Load and preprocess all .mat files from a directory.

    Args:
        data_dir: Directory containing .mat files
        downsample_factor: Decimation factor (default: 5)
        clip_start_sec: Start time for clipping in seconds (default: 3.0)
        clip_end_sec: End time for clipping in seconds (default: 13.0)

    Returns:
        all_data: JumpingData object with concatenated data from all files
    """
    data_dir = Path(data_dir)
    mat_files = sorted(data_dir.glob("*.mat"))

    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    all_X, all_Y = [], []

    for mat_file in mat_files:
        print(f"Loading {mat_file.name}...")
        mat_dict = scipy.io.loadmat(mat_file)

        jd = JumpingData(mat_file.name)
        jd.load_from_dict(mat_dict, downsample_factor=downsample_factor)

        # Clip to remove transients
        hz = 100  # 100 Hz after downsampling
        jd.clip(clip_start_sec, clip_end_sec, hz)

        # Get X, Y pairs
        X, Y = jd.get_XY()
        all_X.append(X)
        all_Y.append(Y)

    # Concatenate all data
    X_combined = np.vstack(all_X)
    Y_combined = np.vstack(all_Y)

    # Create combined JumpingData object
    all_data = JumpingData("combined")
    all_data.X = X_combined
    all_data.Y = Y_combined

    print(f"Loaded {len(mat_files)} files, {X_combined.shape[0]} samples total")

    return all_data
