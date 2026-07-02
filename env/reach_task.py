import numpy as np
from typing import Any, Dict
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two goal arrays (batched)."""
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


class HouseholdReachTask(Task):
    """Reach task inside the NeuroGenesis household room.

    The agent must move its end-effector to a randomly sampled target
    position somewhere in the room.  The achieved goal is the EEF
    position; dense reward = −distance(EEF, target).

    The scene reuses the same rich room layout from custom_task.py
    (walls, furniture, clutter) so the visual context stays consistent
    across all project phases.
    """

    # Room geometry (matches custom_task.py)
    ROOM_HALF = 2.0
    WALL_HEIGHT = 0.6
    WALL_THICKNESS = 0.02

    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "dense",
        distance_threshold: float = 0.05,
        goal_range: float = 1.5,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        # Goals sampled in a box around the origin; z kept low (floor-level tasks)
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0.02])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0.35])
        with self.sim.no_rendering():
            self._create_scene()

    # ------------------------------------------------------------------ #
    #  Scene                                                               #
    # ------------------------------------------------------------------ #
    def _create_scene(self) -> None:
        R = self.ROOM_HALF
        WH = self.WALL_HEIGHT
        WT = self.WALL_THICKNESS

        # ---- Floor ----
        self.sim.create_box(
            body_name="floor",
            half_extents=np.array([R + WT, R + WT, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, -0.01]),
            rgba_color=np.array([0.82, 0.78, 0.72, 1.0]),
            specular_color=np.array([0.3, 0.3, 0.3]),
        )

        # ---- Walls ----
        wall_color = np.array([0.92, 0.91, 0.88, 1.0])
        self.sim.create_box(
            body_name="wall_north",
            half_extents=np.array([R + WT, WT, WH / 2]),
            mass=0.0,
            position=np.array([0.0, R, WH / 2]),
            rgba_color=wall_color,
        )
        self.sim.create_box(
            body_name="wall_south",
            half_extents=np.array([R + WT, WT, WH / 2]),
            mass=0.0,
            position=np.array([0.0, -R, WH / 2]),
            rgba_color=wall_color,
        )
        self.sim.create_box(
            body_name="wall_east",
            half_extents=np.array([WT, R + WT, WH / 2]),
            mass=0.0,
            position=np.array([R, 0.0, WH / 2]),
            rgba_color=wall_color,
        )
        self.sim.create_box(
            body_name="wall_west",
            half_extents=np.array([WT, R + WT, WH / 2]),
            mass=0.0,
            position=np.array([-R, 0.0, WH / 2]),
            rgba_color=wall_color,
        )

        # ---- Kitchen counter ----
        counter_h = 0.35
        self.sim.create_box(
            body_name="kitchen_counter",
            half_extents=np.array([0.8, 0.25, counter_h / 2]),
            mass=0.0,
            position=np.array([0.0, R - 0.27, counter_h / 2]),
            rgba_color=np.array([0.35, 0.25, 0.18, 1.0]),
            specular_color=np.array([0.15, 0.15, 0.15]),
        )

        # ---- Dining table ----
        table_h = 0.30
        self.sim.create_box(
            body_name="dining_table",
            half_extents=np.array([0.40, 0.30, table_h / 2]),
            mass=0.0,
            position=np.array([1.0, -0.5, table_h / 2]),
            rgba_color=np.array([0.55, 0.35, 0.20, 1.0]),
            specular_color=np.array([0.1, 0.1, 0.1]),
        )
        leg_ext = np.array([0.02, 0.02, table_h / 2])
        leg_color = np.array([0.45, 0.30, 0.18, 1.0])
        for i, (lx, ly) in enumerate([
            (1.0 - 0.35, -0.5 - 0.25),
            (1.0 + 0.35, -0.5 - 0.25),
            (1.0 - 0.35, -0.5 + 0.25),
            (1.0 + 0.35, -0.5 + 0.25),
        ]):
            self.sim.create_box(
                body_name=f"table_leg_{i}",
                half_extents=leg_ext,
                mass=0.0,
                position=np.array([lx, ly, table_h / 2]),
                rgba_color=leg_color,
            )

        # ---- Shelf ----
        shelf_h = 0.45
        self.sim.create_box(
            body_name="shelf",
            half_extents=np.array([0.15, 0.50, shelf_h / 2]),
            mass=0.0,
            position=np.array([-R + 0.17, 0.0, shelf_h / 2]),
            rgba_color=np.array([0.70, 0.65, 0.55, 1.0]),
        )

        # ---- Target sphere (visual marker) ----
        self.sim.create_sphere(
            body_name="target",
            radius=0.03,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.9, 0.2, 0.2, 0.6]),
        )

    # ------------------------------------------------------------------ #
    #  Task interface                                                      #
    # ------------------------------------------------------------------ #
    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation for reach

    def get_achieved_goal(self) -> np.ndarray:
        return np.array(self.get_ee_position())

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        return self.np_random.uniform(self.goal_range_low, self.goal_range_high)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
