# Hopper Flying Bug - Fix Instructions

## Problem Summary

The hopper flies upward and off-screen instead of sustained hopping due to:

1. **Hover bias dominates**: `F_hover/4 = 0.00282 N/wing` is added to CEM outputs
2. **CEM range too small**: CEM optimizes `[0, 2e-5 N]` but actual forces are `~0.00282 N` (141× mismatch)
3. **z_des too high**: Target height is 0.20m (20 cm) - unrealistic for hopping
4. **No ground detection**: Forces applied continuously with no awareness of ground contact
5. **No upper bounds**: After adding hover bias, forces have no maximum limit

## Quick Fix - Notebook Cell Changes

### Option 1: Minimal Changes (Quick Fix)

**Cell with `z_des` parameter (around line 917):**
```python
# OLD:
z_des=0.20, w_up=6.0, w_z=5.0, w_u=2e-4,

# NEW:
z_des=0.08, w_up=6.0, w_z=5.0, w_u=2e-4,  # Lowered from 0.20m to 0.08m
```

**Cell with `FMAX_WING` (around line 855 and 3005):**
```python
# OLD:
FMAX_WING = 2.0e-5      # <-- per-wing max (N)

# NEW:
FMAX_WING = 0.003       # 3 mN per wing (realistic range)
```

**Cell with hover bias application (around line 3451):**
```python
# OLD:
F_hover = MASS * 9.81
f_hover = (F_hover / 4.0)
f_vec = np.clip(f_vec_raw + f_hover, 0.0, np.inf)

# NEW - Option A: Reduce hover bias to 50%
F_hover = MASS * 9.81 * 0.5  # Only 50% hover compensation
f_hover = (F_hover / 4.0)
f_vec = np.clip(f_vec_raw + f_hover, 0.0, 0.003)  # Add upper bound

# NEW - Option B: Remove hover bias entirely (let CEM handle it)
# f_hover = 0.0  # No hover bias
# f_vec = np.clip(f_vec_raw, 0.0, 0.003)
```

### Option 2: Complete Fix (Recommended)

Use the provided [hopper_mpc_fixed.py](hopper_mpc_fixed.py) module with proper:
- Ground contact detection
- Adaptive z_des (lower when on ground, higher when airborne)
- Force scaling based on contact state
- Proper CEM force range

**Integration in notebook:**

```python
# Add at top of notebook
from hopper_mpc_fixed import (
    mpc_control_step_fixed,
    cem_optimize_fixed,
    detect_ground_contact,
    Z_DES_HOPPING,
    Z_DES_GROUND,
    FMAX_WING
)

# Replace the mpc_control_step call in your simulation loop:
# OLD:
# last_f, tau_body, R_match = mpc_control_step(model, data, net, R_MPC_1)

# NEW:
last_f, tau_body, is_grounded = mpc_control_step_fixed(
    model, data, net, R_MPC_1,
    apply_forces=True,
    verbose=False  # Set True for debugging
)
```

## Parameter Comparison

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `z_des` | 0.20 m | 0.08 m | Realistic hopping height |
| `FMAX_WING` | 2e-5 N | 0.003 N | Match actual force range |
| Hover bias | 100% weight | 0-50% weight | Let CEM control forces |
| Upper bound | None | 0.003 N | Safety limit |
| Ground detection | None | Added | Adaptive control |

## Expected Behavior After Fix

✓ Hopper jumps to ~8 cm height (adjustable via `Z_DES_HOPPING`)
✓ Returns to ground between jumps
✓ Maintains upright attitude (minimal roll/pitch)
✓ Sustained hopping cycles without flying away
✓ Forces scale down when detecting ground contact

## Debugging Tips

If hopper still misbehaves:

1. **Still flying too high?**
   - Lower `Z_DES_HOPPING` further (try 0.05 m)
   - Reduce `w_z` weight (try 2.0 instead of 5.0)
   - Increase `w_u` to penalize large forces (try 1e-3)

2. **Not jumping at all?**
   - Increase `FMAX_WING` (try 0.005 N)
   - Increase hover bias back to 75-100%
   - Check if `force_scale` is too low

3. **Unstable (tips over)?**
   - Increase `w_up` weight (try 10.0)
   - Increase `w_omega` weight (try 5.0)

4. **Monitor forces during simulation:**
```python
last_f, tau_body, is_grounded = mpc_control_step_fixed(
    model, data, net, R_MPC_1,
    verbose=True  # Prints z, grounded status, forces
)
print(f"Forces: {last_f*1000} mN")  # Convert to millinewtons
```

## Testing Checklist

- [ ] Hopper reaches target height (~8 cm)
- [ ] Returns to ground between jumps
- [ ] Maintains upright orientation
- [ ] Sustained hopping for 5+ seconds
- [ ] No drift in x/y directions
- [ ] Forces stay within bounds [0, 3 mN]

## Additional Improvements

For the notebook itself:
1. Add markdown cells explaining each section
2. Move parameters to top configuration cell
3. Extract functions to separate `.py` files
4. Add validation plots (rollout accuracy, cost convergence)
5. Log states/forces during simulation for analysis
