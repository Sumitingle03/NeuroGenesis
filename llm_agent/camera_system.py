"""Camera System for NeuroGenesis Robot.

Provides live camera feeds from the simulation:
  - Arm Camera: Mounted on the upper arm (link 4), looking down toward the gripper
  - CCTV Camera: Fixed overhead security camera in the corner of the room
"""

import pybullet as p
import numpy as np
import cv2


class CameraSystem:
    # Render every Nth call to avoid GPU bottleneck
    FRAME_SKIP = 5

    def __init__(self, env):
        self.env = env
        self.pc = env.task.sim.physics_client
        self.robot_body_id = env.task.sim._bodies_idx[env.robot.body_name]
        self.ee_link_idx = 8  # panda_hand

        # Smaller resolution = much faster rendering
        self.width = 320
        self.height = 240
        self.fov = 70
        self.aspect = self.width / self.height
        self.near = 0.04
        self.far = 10.0

        self.proj_matrix = self.pc.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far
        )

        # Overhead CCTV — fixed in corner, looking at the room center
        self.overhead_view_matrix = self.pc.computeViewMatrix(
            cameraEyePosition=[2.5, 0.0, 2.8],
            cameraTargetPosition=[0.0, 0.0, 0.0],
            cameraUpVector=[0, 0, 1],
        )

        self._frame_counter = 0
        self._cached_dashboard = None

    # ──────────────────────────────────────────────
    def get_arm_frame(self):
        """Camera mounted on the robot's gripper, looking forward/down."""
        link_state = self.pc.getLinkState(self.robot_body_id, self.ee_link_idx)
        link_pos = np.array(link_state[0])
        link_orn = link_state[1]

        rot = np.array(self.pc.getMatrixFromQuaternion(link_orn)).reshape(3, 3)

        # Position camera near the palm, looking down the fingers (local Z)
        camera_pos = np.array(link_pos) + rot.dot([0, 0, 0.05])
        target_pos = camera_pos + rot.dot([0, 0, 1.0])
        up_vector = rot.dot([0, -1, 0])

        view_matrix = self.pc.computeViewMatrix(camera_pos, target_pos, up_vector)

        _, _, rgb, _, _ = self.pc.getCameraImage(
            self.width, self.height, view_matrix, self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return cv2.cvtColor(
            np.reshape(rgb, (self.height, self.width, 4)).astype(np.uint8),
            cv2.COLOR_RGBA2BGR,
        )

    # ──────────────────────────────────────────────
    def get_overhead_frame(self):
        """Fixed CCTV camera in the corner of the room."""
        _, _, rgb, _, _ = self.pc.getCameraImage(
            self.width, self.height, self.overhead_view_matrix, self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return cv2.cvtColor(
            np.reshape(rgb, (self.height, self.width, 4)).astype(np.uint8),
            cv2.COLOR_RGBA2BGR,
        )

    # ──────────────────────────────────────────────
    def render_dashboard(self):
        """Render the dual-camera dashboard, skipping frames for performance."""
        self._frame_counter += 1

        if self._frame_counter % self.FRAME_SKIP != 0 and self._cached_dashboard is not None:
            # Re-show the last cached frame (very cheap)
            cv2.imshow("NeuroGenesis Vision", self._cached_dashboard)
            cv2.waitKey(1)
            return

        arm_img = self.get_arm_frame()
        cctv_img = self.get_overhead_frame()

        # Labels
        cv2.putText(arm_img, "ARM CAM", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(cctv_img, "CCTV", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        dashboard = np.hstack((arm_img, cctv_img))
        self._cached_dashboard = dashboard

        cv2.imshow("NeuroGenesis Vision", dashboard)
        cv2.waitKey(1)
