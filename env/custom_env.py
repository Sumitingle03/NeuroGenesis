import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from env.mobile_panda import MobilePanda
from env.custom_task import CustomHouseholdTask
from panda_gym.pybullet import PyBullet


class CustomHouseholdPandaEnv(RobotTaskEnv):
    """Custom Pick and Place environment with multi-stage reward shaping.

    Reward = -(distance_EEF_to_object) - (distance_object_to_goal)

    This gives the agent an immediate learning signal for approaching
    the object (stage 1) before it discovers how to grasp and carry
    it to the goal (stage 2).
    """

    def __init__(self, render_mode="rgb_array", reward_type="dense", control_type="ee"):
        sim = PyBullet(render_mode=render_mode, background_color=np.array([40, 45, 50]))
        robot = MobilePanda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomHouseholdTask(sim, reward_type=reward_type)
        super().__init__(robot, task)
        # Store reference for shaped reward
        self._get_ee_position = robot.get_ee_position
        
        # Boost friction on the Panda gripper finger tips for reliable grasping
        robot_id = sim._bodies_idx[robot.body_name]
        for finger_joint in [9, 10]:  # left and right finger links
            sim.physics_client.changeDynamics(
                robot_id, finger_joint,
                lateralFriction=1.5,
                spinningFriction=0.5,
                rollingFriction=0.01,
            )

        # ── Visual Enhancements ──────────────────────────────
        if render_mode == "human":
            import pybullet as p
            pc = sim.physics_client
            # Enable real-time shadows
            pc.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            # Set shadow/light direction (warm afternoon sunlight from above-right)
            pc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide debug panels
            # Set a nice default camera angle
            pc.resetDebugVisualizerCamera(
                cameraDistance=3.5,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0.0, 0.0, 0.15],
            )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Multi-stage dense reward shaping
        ee_pos = np.array(self._get_ee_position())
        obj_pos = obs["achieved_goal"]   # object position
        goal_pos = obs["desired_goal"]   # target position

        dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # Stage 1: approach the object   (weight 1.0)
        # Stage 2: move object to goal   (weight 1.0)
        shaped_reward = -(dist_ee_to_obj + dist_obj_to_goal)

        return obs, np.float32(shaped_reward), terminated, truncated, info

