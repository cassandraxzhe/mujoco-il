"""Quick physics sanity checks for assets/hopper.xml.

Two tests:
  1. Drop with no control -> expect the torso to fall, contact the floor, and bounce.
  2. Apply equal max thrust to all four wings -> expect the torso to lift off.
"""

import numpy as np
import mujoco

XML = "assets/hopper.xml"
BODY = "hopper"


def load():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def rollout(m, d, ctrl, duration):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BODY)
    n = int(duration / m.opt.timestep)
    zs = np.empty(n, dtype=np.float64)
    ts = np.empty(n, dtype=np.float64)
    for i in range(n):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        zs[i] = d.xpos[bid][2]
        ts[i] = d.time
    return ts, zs


def test_drop():
    m, d = load()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BODY)
    z0 = d.xpos[bid][2]

    ctrl = np.zeros(m.nu)
    ts, zs = rollout(m, d, ctrl, duration=1.0)

    z_min = zs.min()
    z_end = zs[-1]
    # bounce: after the minimum we should see z come back up by a detectable amount
    i_min = int(np.argmin(zs))
    z_peak_after = zs[i_min:].max()
    bounce_height = z_peak_after - z_min

    print(f"[drop] z0={z0:.4f}  z_min={z_min:.4f}  z_peak_after_min={z_peak_after:.4f}  "
          f"bounce={bounce_height*1000:.2f} mm  z_end={z_end:.4f}")

    assert z_min < z0, f"robot did not fall (z_min={z_min}, z0={z0})"
    assert bounce_height > 1e-4, f"no detectable bounce (height={bounce_height})"
    print("[drop] PASS")


def test_equal_lift():
    m, d = load()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BODY)
    z0 = d.xpos[bid][2]

    # max per-wing thrust = 0.003 N; total = 12 mN; weight ~ 9.6 mN -> must lift
    ctrl = np.zeros(m.nu)
    for name in ("f1", "f2", "f3", "f4"):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        ctrl[aid] = 0.003

    ts, zs = rollout(m, d, ctrl, duration=1.0)
    z_max = zs.max()
    print(f"[lift] z0={z0:.4f}  z_max={z_max:.4f}  gain={1000*(z_max - z0):.2f} mm")
    assert z_max > z0 + 1e-3, f"robot did not lift off (z_max={z_max}, z0={z0})"
    print("[lift] PASS")


if __name__ == "__main__":
    test_drop()
    test_equal_lift()
