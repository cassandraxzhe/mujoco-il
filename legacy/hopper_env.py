import mujoco
import mujoco.viewer
import numpy as np
import mediapy as media

class HopperEnv:
    def __init__(self, xml_path, render=False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render = render
        self.frames = []
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer
            # self.viewer.launch(self.model, self.data)

        self.renderer =  mujoco.Renderer(self.model)
        self.duration = 5.0
        self.framerate = 60

        # Action = wing thrusts (f1-f4) + leg actuator (leg_actuator)
        self.action_dim = self.model.nu  # number of actuators
        # Observation = pos + vel + orientation (you can customize this)
        self.obs_dim = self.model.nq + self.model.nv  

        # For accumulating contact impulses
        self._impulse_accumulator = 0.0



    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs

    def step(self, action):
        # clip actions to actuator limits
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Update contact impulse accumulator
        self._accumulate_impulse()

        obs = self._get_obs()
        reward = 0.0   # you can define a reward if needed
        done = False   # or implement termination condition
        info = {}

        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        if self.render and self.viewer is not None:
            if self.data.time < self.duration:
                # print(f"{self.data.time} < {self.duration}")
                if len(self.frames) < self.data.time * self.framerate:
                    # print("adding another frame")
                    self.renderer.update_scene(self.data, scene_option=scene_option)
                    pixels = self.renderer.render()
                    self.frames.append(pixels)

        #     self.viewer.sync()

        return obs, reward, done, info
    
    def simulate_and_render(self, model, sim_time=5.0): #, iters=10000):

        # self.frames = []

        # scene_option = mujoco.MjvOption()
        # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        # for _ in range(iters):
        #     action = np.random.uniform(-1, 1, size=self.action_dim)  # random actions TODO: uh change this later?
        #     obs, reward, done, info = self.step(action)
        #     # print("Impulse so far:", env._impulse_accumulator)
        #     if len(self.frames) < self.data.time * self.framerate:
        #         # print("adding another frame")
        #         self.renderer.update_scene(self.data, scene_option=scene_option)
        #         pixels = self.renderer.render()
        #         self.frames.append(pixels)

        # print(len(self.frames))
        # media.show_video(self.frames, fps=self.framerate)

        """
        Run the hopper simulation with MPC + learned dynamics and render it.
        
        Args:
            env: MuJoCo environment (must have env.physics)
            model: trained dynamics model
            sim_time: total simulation time in seconds
        """
        t = 0.0
        viewer = mujoco.viewer.launch_passive(self.physics)  # opens interactive window
        while t < sim_time:
            # Get current hopper state
            pos, eul = mj_get_state(self)

            # Compute MPC control
            u = cem_optimize(model, pos, eul)   # or use mpc_control_step(env, model)
            mj_apply_u(self, u)

            # Step physics
            substeps = max(1, int(DT / env.physics.model.opt.timestep + 1e-9))
            for _ in range(substeps):
                env.step()

            # Render
            viewer.render()

            # Optional: slow down for real-time visualization
            time.sleep(DT)

            # Advance simulation time
            t += DT

        viewer.close()

    def _get_obs(self):
        # # Example: positions + velocities
        # return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
        # Example: positions, velocities, and current impulse accumulator
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            np.array([self._impulse_accumulator])
        ])
    
    def _accumulate_impulse(self):
        """Accumulate vertical contact impulse on the leg."""
        # Reset accumulator if no contact (start of flight phase)
        if self.data.ncon == 0:
            self._impulse_accumulator = 0.0
            return

        # Loop over contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Contact force (6D: force + torque) for body in world frame
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)

            # Take vertical component (z-axis, assuming ground is z-up)
            vertical_force = c_array[2]

            # Integrate force over dt to get impulse
            self._impulse_accumulator += vertical_force * self.model.opt.timestep
    
    def video(self):
        # print(len(self.frames))
        # media.show_video(self.frames, fps=self.framerate)
        self.viewer.launch(self.model, self.data)