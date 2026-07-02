import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from env.mobile_panda import MobilePanda
from env.reach_task import HouseholdReachTask


class NeuroGenesisReachEnv(RobotTaskEnv):
    """Mobile manipulator tasked with reaching a target position.

    The gripper is locked (block_gripper=True) since this phase only
    trains navigation + reaching, not grasping.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "ee",
    ):
        sim = PyBullet(render_mode=render_mode)
        robot = MobilePanda(
            sim,
            block_gripper=True,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )
        task = HouseholdReachTask(
            sim,
            get_ee_position=robot.get_ee_position,
            reward_type=reward_type,
        )
        super().__init__(
            robot,
            task,
            render_distance=3.0,
            render_yaw=45,
            render_pitch=-25,
        )
