"""
Neural network models for hopper dynamics learning.
"""

import torch
import torch.nn as nn


class HopperMLP(nn.Module):
    """
    Multi-Layer Perceptron for learning hopper forward dynamics.

    Maps current state + action to next state change (delta prediction).

    Args:
        input_dim: Input dimension (typically 14: pos(3) + eul(3) + thrust(1) + tau(3) + signals(4))
        output_dim: Output dimension (typically 6: delta_pos(3) + delta_eul(3))
        hidden_dim: Hidden layer size (default: 32)

    Example:
        >>> model = HopperMLP(input_dim=14, output_dim=6, hidden_dim=32)
        >>> x = torch.randn(32, 14)  # batch_size=32
        >>> delta = model(x)  # Output: [32, 6]
    """

    def __init__(self, input_dim=14, output_dim=6, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            delta: Predicted state change [batch_size, output_dim]
        """
        return self.net(x)

    def save(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=device))
        return self
