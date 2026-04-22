# Quick Fix for NameError: name 'time' is not defined

## The Error

```
NameError: name 'time' is not defined
```

This occurs because Cell 23 uses `time.time()` but doesn't import the `time` module.

## Solution

**Add this import at the beginning of Cell 23** (before the function definitions):

```python
import time
import numpy as np
import torch
import mujoco
```

## Where to Add It

Look for the cell that contains `def run_closed_loop_and_record(...)` and add `import time` at the top of that cell.

The cell should start like this:

```python
import time  # <-- ADD THIS LINE
import numpy as np
import torch
import mujoco

# ... rest of the code ...

def run_closed_loop_and_record(hopper_xml=HOPPER_XML, model_weights=MODEL_WEIGHTS,
                               sim_time=6.0, fps=80, camera_name=None):
    # ...
    t0 = time.time()  # <-- This line needs the import above
    # ...
```

## Alternative: Add to Main Imports Cell

If you prefer, you can add `import time` to your main imports cell at the top of the notebook (the one that has `import mujoco`, `import scipy.io`, etc.).

Then restart the kernel and run all cells in order.
