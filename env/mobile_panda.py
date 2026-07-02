import numpy as np
from gymnasium import spaces
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from typing import Optional


class MobilePanda(Panda):
    """Panda arm mounted on a Roomba-like cylindrical mobile base.

    The base is a flat disc (cylinder) that slides along the floor.
    Two extra action dimensions control base dx and dy.
    """

    ROOMBA_RADIUS = 0.17  # iRobot Roomba is ~34 cm diameter
    ROOMBA_HEIGHT = 0.09  # ~9 cm tall
    ROOMBA_COLOR = np.array([0.20, 0.22, 0.25, 1.0])  # dark charcoal
    ROOMBA_ACCENT = np.array([0.35, 0.75, 0.55, 1.0])  # teal-green accent ring
    BASE_SPEED = 0.05  # max displacement per step

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        # Raise the Panda arm so it sits on top of the Roomba disc
        base_position = base_position if base_position is not None else np.zeros(3)
        # Lift the arm base by Roomba height so it sits on top
        base_position = base_position.copy()
        base_position[2] = self.ROOMBA_HEIGHT

        super().__init__(sim, block_gripper, base_position, control_type)

        # Build the action space: 2 (base) + 3 or 7 (arm) + 0 or 1 (gripper)
        n_arm = 3 if self.control_type == "ee" else 7
        n_grip = 0 if self.block_gripper else 1
        self.n_action = 2 + n_arm + n_grip
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_action,), dtype=np.float32)

        # Create visual Roomba base body
        self._create_roomba_base(base_position)

    # ------------------------------------------------------------------ #
    #  Roomba visual                                                      #
    # ------------------------------------------------------------------ #
    def _create_roomba_base(self, base_position: np.ndarray) -> None:
        """Spawn the Roomba-like disc underneath the arm."""
        # Main body disc
        self.sim.create_cylinder(
            body_name="roomba_body",
            radius=self.ROOMBA_RADIUS,
            height=self.ROOMBA_HEIGHT,
            mass=0.0,
            ghost=True,
            position=np.array([base_position[0], base_position[1], self.ROOMBA_HEIGHT / 2]),
            rgba_color=self.ROOMBA_COLOR,
        )
        # Accent ring (slightly larger, very thin)
        self.sim.create_cylinder(
            body_name="roomba_ring",
            radius=self.ROOMBA_RADIUS + 0.005,
            height=0.015,
            mass=0.0,
            ghost=True,
            position=np.array([base_position[0], base_position[1], self.ROOMBA_HEIGHT - 0.005]),
            rgba_color=self.ROOMBA_ACCENT,
        )
        # Front sensor bump (small sphere)
        self.sim.create_sphere(
            body_name="roomba_bumper",
            radius=0.025,
            mass=0.0,
            ghost=True,
            position=np.array([
                base_position[0] + self.ROOMBA_RADIUS * 0.7,
                base_position[1],
                self.ROOMBA_HEIGHT / 2,
            ]),
            rgba_color=np.array([0.12, 0.12, 0.15, 1.0]),
        )

    def _sync_roomba_visual(self, new_x: float, new_y: float) -> None:
        """Move every Roomba visual part to follow the arm base."""
        ori = [0.0, 0.0, 0.0, 1.0]
        pc = self.sim.physics_client
        idx = self.sim._bodies_idx

        pc.resetBasePositionAndOrientation(
            idx["roomba_body"],
            [new_x, new_y, self.ROOMBA_HEIGHT / 2],
            ori,
        )
        pc.resetBasePositionAndOrientation(
            idx["roomba_ring"],
            [new_x, new_y, self.ROOMBA_HEIGHT - 0.005],
            ori,
        )
        pc.resetBasePositionAndOrientation(
            idx["roomba_bumper"],
            [new_x + self.ROOMBA_RADIUS * 0.7, new_y, self.ROOMBA_HEIGHT / 2],
            ori,
        )

    # ------------------------------------------------------------------ #
    #  Action handling                                                     #
    # ------------------------------------------------------------------ #
    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # --- Base movement (first 2 dims) ---
        base_ctrl = action[:2] * self.BASE_SPEED
        pc = self.sim.physics_client
        body_id = self.sim._bodies_idx[self.body_name]
        cur_pos, cur_ori = pc.getBasePositionAndOrientation(body_id)

        new_x = cur_pos[0] + base_ctrl[0]
        new_y = cur_pos[1] + base_ctrl[1]
        pc.resetBasePositionAndOrientation(
            body_id, [new_x, new_y, cur_pos[2]], cur_ori
        )

        # Keep visual Roomba in sync
        self._sync_roomba_visual(new_x, new_y)

        # --- Arm + gripper (remaining dims) ---
        arm_action = action[2:]
        if self.control_type == "ee":
            target_arm_angles = self.ee_displacement_to_target_arm_angles(arm_action[:3])
        else:
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_action[:7])

        if self.block_gripper:
            target_fingers_width = 0.0
        else:
            fingers_ctrl = arm_action[-1] * 0.2
            target_fingers_width = self.get_fingers_width() + fingers_ctrl

        target_angles = np.concatenate(
            (target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2])
        )
        self.control_joints(target_angles=target_angles)

    # ------------------------------------------------------------------ #
    #  Observations                                                        #
    # ------------------------------------------------------------------ #
    def get_obs(self) -> np.ndarray:
        pc = self.sim.physics_client
        body_id = self.sim._bodies_idx[self.body_name]
        base_pos, _ = pc.getBasePositionAndOrientation(body_id)
        base_xy = np.array(base_pos[:2])
        arm_obs = super().get_obs()
        return np.concatenate((base_xy, arm_obs))

    def reset(self) -> None:
        super().reset()
        # Re-sync Roomba visual on episode reset
        pc = self.sim.physics_client
        body_id = self.sim._bodies_idx[self.body_name]
        pos, _ = pc.getBasePositionAndOrientation(body_id)
        self._sync_roomba_visual(pos[0], pos[1])
