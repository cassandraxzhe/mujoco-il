"""
MuJoCo simulation and control loop for hopper.
"""

import os
import time
import numpy as np
import torch
import mujoco
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

from .models import HopperMLP
from .mpc import cem_optimize
from .frame_alignment import (
    build_R_match,
    body_to_nn,
    tau_nn_to_body,
    allocate_per_wing_forces,
    get_body_rotation_matrix
)


# ============================================================================
# MUJOCO HELPER FUNCTIONS
# ============================================================================

def body_name_to_id(mjmodel, name):
    """
    Get MuJoCo body ID by name.

    Args:
        mjmodel: MuJoCo model
        name: Body name string

    Returns:
        body_id: Integer body ID
    """
    return mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, name)


def actuator_name_to_id(mjmodel, name):
    """
    Get MuJoCo actuator ID by name.

    Args:
        mjmodel: MuJoCo model
        name: Actuator name string

    Returns:
        actuator_id: Integer actuator ID
    """
    return mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def get_body_pos_eul(mjmodel, mjdata, body_name='hopper'):
    """
    Extract body position and Euler angles (roll, pitch, yaw).

    Args:
        mjmodel: MuJoCo model
        mjdata: MuJoCo data
        body_name: Name of body (default: 'hopper')

    Returns:
        pos: [3] position (x, y, z) in meters
        eul: [3] euler angles (roll, pitch, yaw) in radians
    """
    bid = body_name_to_id(mjmodel, body_name)
    pos = np.array(mjdata.xpos[bid], dtype=np.float32)

    # Extract rotation matrix and convert to Euler angles
    R_all = np.asarray(mjdata.xmat).reshape(-1, 3, 3)
    R = R_all[bid]

    # ZYX Euler convention (roll, pitch, yaw)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    eul = np.array([roll, pitch, yaw], dtype=np.float32)
    return pos, eul


def apply_ctrl_to_data(mjmodel, mjdata, u_vec, actuator_names=('f1', 'f2', 'f3', 'f4')):
    """
    Apply per-wing forces to MuJoCo actuators.

    Args:
        mjmodel: MuJoCo model
        mjdata: MuJoCo data
        u_vec: [4] per-wing forces (N)
        actuator_names: Tuple of actuator names (default: f1, f2, f3, f4)
    """
    idxs = [actuator_name_to_id(mjmodel, name) for name in actuator_names]
    for i, aidx in enumerate(idxs):
        mjdata.ctrl[aidx] = float(u_vec[i])


def get_system_com(mjdata):
    """
    Get system center of mass position.

    Args:
        mjdata: MuJoCo data

    Returns:
        com: [3] system COM position (x, y, z)
    """
    return np.array(mjdata.subtree_com[0])


# ============================================================================
# SIMULATION LOOP
# ============================================================================

def run_simulation(
    hopper_xml="flying_hopper.xml",
    model_weights=None,
    sim_time=5.0,
    fps=80,
    body_name='hopper',
    z_des=0.20,
    output_video="hopper_sim.mp4",
    device='cpu',
    verbose=True,
    plot=True,
    use_frame_realignment=True
):
    """
    Run closed-loop MPC simulation with video rendering and COM tracking.

    Args:
        hopper_xml: Path to MuJoCo XML model file
        model_weights: Path to trained PyTorch model weights (.pt file)
        sim_time: Simulation duration in seconds
        fps: Video frames per second
        body_name: Name of hopper body in MuJoCo model
        z_des: Desired height for MPC cost function
        output_video: Output video filename
        device: PyTorch device ('cpu' or 'cuda')
        verbose: Print debug information
        plot: Show COM trajectory plots
        use_frame_realignment: Use frame realignment for orientation-robust control

    Returns:
        results: Dictionary containing:
            - 'frames': List of rendered video frames
            - 'torso_com': Dict with 't', 'x', 'y', 'z' arrays
            - 'system_com': Dict with 't', 'x', 'y', 'z' arrays
            - 'controls': List of control inputs
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Hopper MPC Simulation")
        print(f"{'='*60}")
        print(f"Model:      {hopper_xml}")
        print(f"Weights:    {model_weights}")
        print(f"Duration:   {sim_time}s @ {fps} fps")
        print(f"z_des:      {z_des:.3f} m")
        print(f"Frame Realignment: {'Enabled' if use_frame_realignment else 'Disabled'}")
        print(f"{'='*60}\n")

    # 1) Load MuJoCo model & data
    mjmodel = mujoco.MjModel.from_xml_path(hopper_xml)
    mjdata = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, mjdata)

    # 2) Create renderer
    width, height = 640, 480
    renderer = mujoco.Renderer(mjmodel, width=width, height=height)

    # 3) Load PyTorch model
    input_dim = 14  # [pos(3), eul(3), F(1), tau(3), signals(4)]
    output_dim = 6  # [delta_pos(3), delta_eul(3)]
    net = HopperMLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=32).to(device)

    if model_weights is not None:
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Model weights not found: {model_weights}")
        net.load_state_dict(torch.load(model_weights, map_location=device))
        if verbose:
            print(f"✓ Loaded model weights from {model_weights}")

    net.eval()

    # 4) Time & frame bookkeeping
    fps = int(fps)
    n_frames = int(sim_time * fps)
    dt_video = 1.0 / fps
    steps_per_frame = max(1, int(round(dt_video / mjmodel.opt.timestep)))

    # 5) Storage
    frames = []
    controls = []
    torso_com = {'t': [], 'x': [], 'y': [], 'z': []}
    system_com = {'t': [], 'x': [], 'y': [], 'z': []}

    simt = 0.0

    # Reference direction for frame realignment (forward = +x in world frame)
    R_mpc_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # 6) Main simulation loop
    for frame_idx in range(n_frames):
        # Get current state
        pos, eul = get_body_pos_eul(mjmodel, mjdata, body_name=body_name)
        bid = body_name_to_id(mjmodel, body_name)

        if use_frame_realignment:
            # Get body rotation matrix
            R_body = get_body_rotation_matrix(mjdata, bid)

            # Build alignment matrix (aligns body frame to NN frame)
            R_match = build_R_match(R_body, R_mpc_ref)

            # Transform rotation to NN frame
            R_nn = R_match.T @ R_body

            # Convert aligned rotation to Euler angles for NN input
            r = Rot.from_matrix(R_nn)
            eul_nn = r.as_euler('xyz', degrees=False).astype(np.float32)

            # Run CEM optimization in NN frame
            t_cem_start = time.time()
            u0_nn = cem_optimize(net, pos, eul_nn, z_des=z_des, device=device, verbose=False)
            t_cem = time.time() - t_cem_start

            # The control output u0_nn is [f1, f2, f3, f4] wing forces
            # These are already per-wing forces in the aligned frame
            u0 = u0_nn
        else:
            # Original: no frame realignment
            t_cem_start = time.time()
            u0 = cem_optimize(net, pos, eul, z_des=z_des, device=device, verbose=False)
            t_cem = time.time() - t_cem_start

        # Apply control
        apply_ctrl_to_data(mjmodel, mjdata, u0)
        controls.append(u0.copy())

        # Step physics
        for _ in range(steps_per_frame):
            mujoco.mj_step(mjmodel, mjdata)

        # Log body COM
        bid = body_name_to_id(mjmodel, body_name)
        body_pos = np.array(mjdata.xpos[bid])
        torso_com['t'].append(simt)
        torso_com['x'].append(body_pos[0])
        torso_com['y'].append(body_pos[1])
        torso_com['z'].append(body_pos[2])

        # Log system COM
        sys_com = get_system_com(mjdata)
        system_com['t'].append(simt)
        system_com['x'].append(sys_com[0])
        system_com['y'].append(sys_com[1])
        system_com['z'].append(sys_com[2])

        # Render frame
        renderer.update_scene(mjdata)
        frame = renderer.render()
        frames.append(frame)

        # Update time
        simt += steps_per_frame * mjmodel.opt.timestep

        # Print progress
        if verbose and (frame_idx % 20 == 0 or frame_idx == n_frames - 1):
            print(f"Frame {frame_idx:3d}/{n_frames} | "
                  f"t={simt:5.2f}s | "
                  f"z={body_pos[2]:6.3f}m | "
                  f"u={np.sum(u0)*1000:6.2f}mN | "
                  f"CEM={t_cem*1000:5.1f}ms")

    # 7) Save video
    if output_video:
        imageio.mimsave(output_video, frames, fps=fps)
        if verbose:
            print(f"\n✓ Saved video: {output_video}")

    # 8) Plot trajectories
    if plot:
        plot_com_trajectories(torso_com, system_com, z_des=z_des)

    # 9) Return results
    results = {
        'frames': frames,
        'torso_com': torso_com,
        'system_com': system_com,
        'controls': controls,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Simulation Complete!")
        print(f"{'='*60}\n")

    return results


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def plot_com_trajectories(torso_com, system_com, z_des=None):
    """
    Plot torso and system COM trajectories.

    Args:
        torso_com: Dict with 't', 'x', 'y', 'z' lists
        system_com: Dict with 't', 'x', 'y', 'z' lists
        z_des: Desired height (optional, draws reference line)
    """
    # Time series plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # X position
    axes[0].plot(torso_com['t'], torso_com['x'], label='Torso X', linewidth=2)
    axes[0].plot(system_com['t'], system_com['x'], '--', label='System X', alpha=0.7)
    axes[0].set_ylabel('X Position [m]')
    axes[0].legend()
    axes[0].grid(True)

    # Y position
    axes[1].plot(torso_com['t'], torso_com['y'], label='Torso Y', linewidth=2)
    axes[1].plot(system_com['t'], system_com['y'], '--', label='System Y', alpha=0.7)
    axes[1].set_ylabel('Y Position [m]')
    axes[1].legend()
    axes[1].grid(True)

    # Z position
    axes[2].plot(torso_com['t'], torso_com['z'], label='Torso Z', linewidth=2)
    axes[2].plot(system_com['t'], system_com['z'], '--', label='System Z', alpha=0.7)
    if z_des is not None:
        axes[2].axhline(z_des, color='r', linestyle=':', label=f'z_des={z_des:.2f}m')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Z Position [m]')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('Hopper COM Trajectory')
    plt.tight_layout()
    plt.show()

    # 3D trajectory plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(torso_com['x'], torso_com['y'], torso_com['z'],
            label='Torso COM', linewidth=2, color='blue')
    ax.plot(system_com['x'], system_com['y'], system_com['z'],
            '--', label='System COM', linewidth=2, alpha=0.7, color='orange')

    # Mark start and end
    ax.scatter([torso_com['x'][0]], [torso_com['y'][0]], [torso_com['z'][0]],
               c='green', s=100, marker='o', label='Start')
    ax.scatter([torso_com['x'][-1]], [torso_com['y'][-1]], [torso_com['z'][-1]],
               c='red', s=100, marker='x', label='End')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Hopper COM Trajectory')
    ax.legend()

    plt.show()


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

# Backward compatibility
get_torso_pos_eul = get_body_pos_eul
run_closed_loop_and_record = run_simulation
