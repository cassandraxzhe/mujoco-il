"""
MPC Configuration Parameters

Centralized configuration for Model Predictive Control.
Import and modify these values as needed for different experiments.
"""

# ============================================================================
# PHYSICAL PARAMETERS
# ============================================================================

# Hopper geometry
L = 0.015  # Arm length (torso center to wing), meters (15 mm)

# Mass properties
MASS_BODY = 0.00095  # kg (0.95 mg)
MASS_LEG = 0.00020   # kg (0.20 mg)
MASS_TOTAL = MASS_BODY + MASS_LEG  # kg (1.15 mg)

# Gravity
G = 9.81  # m/s²

# Hover force (total upward force needed to counteract gravity)
F_HOVER = MASS_TOTAL * G  # ~0.01128 N (11.28 mN)


# ============================================================================
# CONTROL PARAMETERS
# ============================================================================

# Force limits
FMAX_WING = 0.003  # Maximum force per wing, Newtons (3 mN)
# NOTE: Original notebook had 2e-5 N in some cells, but 0.003 N is more realistic

# Control frequency
CTRL_FREQ = 80  # Hz
DT = 1.0 / CTRL_FREQ  # Control timestep (0.0125 s)

# Control dimension
CTRL_DIM = 4  # Per-wing forces: [f1, f2, f3, f4]


# ============================================================================
# MPC PARAMETERS
# ============================================================================

# Horizon
H = 20  # MPC horizon steps (~0.25s at 80 Hz)

# CEM optimization
POP = 256       # CEM population size
ELITES = 32     # Number of elite samples
ITERS = 4       # CEM iterations per MPC step


# ============================================================================
# COST FUNCTION WEIGHTS
# ============================================================================

# Target state
Z_DES = 0.20  # Desired height, meters (20 cm)
# NOTE: For sustained hopping, consider lowering to 0.08 m

# Cost weights
W_UP = 6.0      # Attitude stability (roll/pitch) - higher = more upright
W_Z = 5.0       # Height tracking - higher = stick closer to z_des
W_XY = 2.0      # Lateral drift penalty - higher = stay in vertical column
W_U = 2e-4      # Control effort - higher = use less force
W_OMEGA = 2.0   # Angular velocity damping - higher = reduce spinning


# ============================================================================
# RECOMMENDED PRESETS
# ============================================================================

class HoppingConfig:
    """Configuration for sustained hopping behavior."""
    Z_DES = 0.08        # Lower target height (8 cm)
    W_UP = 6.0
    W_Z = 5.0
    W_XY = 2.0
    W_U = 2e-4
    W_OMEGA = 2.0
    FMAX_WING = 0.003   # 3 mN per wing


class HighJumpConfig:
    """Configuration for higher jumping."""
    Z_DES = 0.20        # Higher target (20 cm)
    W_UP = 8.0          # More attitude stability needed
    W_Z = 7.0           # Stronger height tracking
    W_XY = 3.0
    W_U = 1e-4          # Allow more control effort
    W_OMEGA = 3.0
    FMAX_WING = 0.005   # 5 mN per wing


class ConservativeConfig:
    """Conservative configuration for stable, low hopping."""
    Z_DES = 0.05        # Very low target (5 cm)
    W_UP = 10.0         # Maximum stability
    W_Z = 3.0           # Softer height tracking
    W_XY = 4.0          # Strong drift penalty
    W_U = 5e-4          # Minimize control effort
    W_OMEGA = 4.0       # Strong damping
    FMAX_WING = 0.002   # 2 mN per wing


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config(config_name='default'):
    """
    Load a predefined configuration.

    Args:
        config_name: 'default', 'hopping', 'high_jump', or 'conservative'

    Returns:
        config: Dictionary of parameters
    """
    if config_name == 'hopping':
        cfg = HoppingConfig
    elif config_name == 'high_jump':
        cfg = HighJumpConfig
    elif config_name == 'conservative':
        cfg = ConservativeConfig
    else:
        # Return default values from this module
        return {
            'Z_DES': Z_DES,
            'W_UP': W_UP,
            'W_Z': W_Z,
            'W_XY': W_XY,
            'W_U': W_U,
            'W_OMEGA': W_OMEGA,
            'FMAX_WING': FMAX_WING,
            'H': H,
            'POP': POP,
            'ELITES': ELITES,
            'ITERS': ITERS,
            'DT': DT,
            'L': L,
        }

    # Extract config class attributes
    return {
        'Z_DES': cfg.Z_DES,
        'W_UP': cfg.W_UP,
        'W_Z': cfg.W_Z,
        'W_XY': cfg.W_XY,
        'W_U': cfg.W_U,
        'W_OMEGA': cfg.W_OMEGA,
        'FMAX_WING': cfg.FMAX_WING,
        'H': H,
        'POP': POP,
        'ELITES': ELITES,
        'ITERS': ITERS,
        'DT': DT,
        'L': L,
    }


def print_config(config_name='default'):
    """Print configuration parameters."""
    cfg = load_config(config_name)
    print(f"\n{'='*50}")
    print(f"MPC Configuration: {config_name.upper()}")
    print(f"{'='*50}")
    print(f"\nTarget State:")
    print(f"  Z_DES:      {cfg['Z_DES']:.3f} m ({cfg['Z_DES']*100:.1f} cm)")
    print(f"\nCost Weights:")
    print(f"  W_UP:       {cfg['W_UP']:.1f}  (attitude)")
    print(f"  W_Z:        {cfg['W_Z']:.1f}  (height)")
    print(f"  W_XY:       {cfg['W_XY']:.1f}  (lateral)")
    print(f"  W_U:        {cfg['W_U']:.1e}  (control effort)")
    print(f"  W_OMEGA:    {cfg['W_OMEGA']:.1f}  (angular velocity)")
    print(f"\nControl Limits:")
    print(f"  FMAX_WING:  {cfg['FMAX_WING']*1000:.2f} mN per wing")
    print(f"  F_HOVER:    {F_HOVER*1000:.2f} mN total")
    print(f"\nMPC Parameters:")
    print(f"  Horizon:    {cfg['H']} steps ({cfg['H']*cfg['DT']:.3f} s)")
    print(f"  Population: {cfg['POP']}")
    print(f"  Elites:     {cfg['ELITES']}")
    print(f"  Iterations: {cfg['ITERS']}")
    print(f"  DT:         {cfg['DT']*1000:.2f} ms ({1/cfg['DT']:.0f} Hz)")
    print(f"{'='*50}\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Available configurations: 'default', 'hopping', 'high_jump', 'conservative'")
    print_config('default')
    print_config('hopping')
    print_config('high_jump')
    print_config('conservative')
