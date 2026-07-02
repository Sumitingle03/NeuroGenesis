import time
import numpy as np
import pybullet as p


class RobotSkills:
    """A library of low-level skills executed by the mobile manipulator.
    
    Uses constraint-based grasping: when the gripper closes near an object,
    a PyBullet fixed constraint attaches the object to the end-effector.
    This is a standard simulation technique used in robotics research.
    """

    # Known surface heights (Z coordinate of the TOP surface)
    SURFACE_HEIGHTS = {
        "kitchen_counter": 0.35,
        "dining_table": 0.30,
        "shelf": 0.45,
        "shelf_board": 0.45 * 0.55 + 0.008,
        "dustbin": 0.40,
        "floor": 0.0,
    }

    # Known furniture approach directions: where the robot should stand
    # relative to the furniture center to reach it.
    APPROACH_OFFSETS = {
        "kitchen_counter": (0.0, -0.40),
        "dining_table":    (-0.55, 0.0),
        "shelf":           (0.55, 0.0),
        "dustbin":         (-0.55, 0.0),
        "tv":              (0.0, 0.55),   # stand north of TV (it's on south wall)
        "lamp":            (-0.45, 0.0),  # stand west of lamp
        "dirt":            (-0.45, 0.0),  # stand west of dirt patch
    }

    # Smart device ON/OFF colors
    DEVICE_COLORS = {
        "tv":   {"off": [0.08, 0.08, 0.08, 1.0], "on": [0.3, 0.6, 1.0, 1.0]},   # black → blue glow
        "lamp": {"off": [0.3, 0.3, 0.25, 1.0],   "on": [1.0, 0.95, 0.4, 1.0]},   # grey → bright yellow
    }

    MAX_STEPS = 500  # Safety limit per skill

    def __init__(self, env, camera_system=None):
        self.env = env
        self.camera_system = camera_system
        self._physics_client = env.task.sim.physics_client
        self.gripper_ctrl = 1.0  # 1.0 = open, -1.0 = closed
        
        # Constraints for grasping
        self._grasp_constraint = None
        self._attached_object = None
        self._grasped_body_id = None
        # Track device on/off state
        self._device_states = {name: False for name in self.DEVICE_COLORS}

    def _p_control(self, current, target, kp=5.0, max_vel=1.0):
        """Proportional control helper."""
        diff = target - current
        vel = kp * diff
        norm = np.linalg.norm(vel)
        if norm > max_vel:
            vel = vel * (max_vel / norm)
        return vel

    def _step(self, action):
        """Execute one step with current gripper state applied."""
        action[5] = self.gripper_ctrl
        obs, _, _, _, _ = self.env.step(action)
        
        # Render live cameras during movement!
        if self.camera_system:
            self.camera_system.render_dashboard()
            
        time.sleep(0.01)
        return obs

    def get_object_position(self, object_name: str) -> np.ndarray:
        """Query the physics engine for the exact position of a named object."""
        if object_name not in self.env.task.sim._bodies_idx:
            raise ValueError(f"Object '{object_name}' not found in the simulation.")
        body_id = self.env.task.sim._bodies_idx[object_name]
        pos, _ = self._physics_client.getBasePositionAndOrientation(body_id)
        return np.array(pos)

    def _attach_object(self, object_name: str):
        """Create a fixed constraint to weld the object to the end-effector."""
        if self._grasp_constraint is not None:
            return  # Already holding something
            
        obj_body_id = self.env.task.sim._bodies_idx[object_name]
        robot_body_id = self.env.task.sim._bodies_idx[self.env.robot.body_name]
        
        # Link 8 is the panda_hand link (the palm between the two fingers)
        ee_link = 8
        
        # Get current positions to compute the relative offset
        ee_state = self._physics_client.getLinkState(robot_body_id, ee_link)
        ee_pos = np.array(ee_state[0])
        obj_pos, obj_orn = self._physics_client.getBasePositionAndOrientation(obj_body_id)
        
        # Offset from EE frame to object
        inv_ee_pos, inv_ee_orn = self._physics_client.invertTransform(ee_state[0], ee_state[1])
        obj_in_ee_pos, obj_in_ee_orn = self._physics_client.multiplyTransforms(
            inv_ee_pos, inv_ee_orn, obj_pos, obj_orn
        )
        
        self._grasp_constraint = self._physics_client.createConstraint(
            parentBodyUniqueId=robot_body_id,
            parentLinkIndex=ee_link,
            childBodyUniqueId=obj_body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=obj_in_ee_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=obj_in_ee_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        # Set high force so object stays firmly attached
        self._physics_client.changeConstraint(self._grasp_constraint, maxForce=100)
        self._grasped_body_id = obj_body_id
        print(f"  [GRASP] Object '{object_name}' attached to gripper via constraint.")

    def _detach_object(self):
        """Remove the fixed constraint to release the object."""
        if self._grasp_constraint is not None:
            self._physics_client.removeConstraint(self._grasp_constraint)
            self._grasp_constraint = None
            self._grasped_body_id = None
            print(f"  [GRASP] Object released.")

    def _get_approach_position(self, target_name: str, target_pos: np.ndarray) -> np.ndarray:
        """Figure out the best XY position for the base to reach the target."""
        # Check if this IS a piece of furniture
        if target_name in self.APPROACH_OFFSETS:
            dx, dy = self.APPROACH_OFFSETS[target_name]
            return np.array([target_pos[0] + dx, target_pos[1] + dy])

        # Check if this object is sitting ON a piece of furniture
        obj_z = target_pos[2]
        parent_furniture = None
        for furn_name, surface_z in self.SURFACE_HEIGHTS.items():
            if furn_name == "floor":
                continue
            if abs(obj_z - surface_z) < 0.25:
                try:
                    furn_pos = self.get_object_position(furn_name)
                    if abs(target_pos[0] - furn_pos[0]) < 1.0 and abs(target_pos[1] - furn_pos[1]) < 1.0:
                        parent_furniture = furn_name
                        break
                except ValueError:
                    continue

        if parent_furniture and parent_furniture in self.APPROACH_OFFSETS:
            dx, dy = self.APPROACH_OFFSETS[parent_furniture]
            return np.array([target_pos[0] + dx, target_pos[1] + dy])

        # Fallback: approach from the robot's current side
        base_xy = self.env.robot.get_obs()[:2]
        diff = base_xy - target_pos[:2]
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            diff = np.array([0.5, 0.0])
            dist = 0.5
        approach_dir = diff / dist
        return target_pos[:2] + approach_dir * 0.5

    def navigate_to(self, target_pos: np.ndarray, target_name: str = ""):
        """Drive the mobile base to a target location while keeping the arm high."""
        approach_xy = self._get_approach_position(target_name, target_pos)
        print(f"  [NAV] Driving base to ({approach_xy[0]:.2f}, {approach_xy[1]:.2f})...")
        
        for _ in range(self.MAX_STEPS):
            ee_pos = self.env.robot.get_ee_position()
            base_xy = self.env.robot.get_obs()[:2]
            action = np.zeros(6, dtype=np.float32)

            action[:2] = self._p_control(base_xy, approach_xy, kp=3.0)
            
            # Keep arm high during transit (especially important when carrying)
            ee_target = np.array([base_xy[0] + 0.3, base_xy[1], 0.7])
            action[2:5] = self._p_control(ee_pos, ee_target, kp=2.0)
            
            self._step(action)

            if np.linalg.norm(approach_xy - base_xy) < 0.05:
                print(f"  [NAV] Arrived!")
                break

    def pick(self, target_pos: np.ndarray, target_name: str = "object"):
        """Hover above → lower arm → attach object via constraint → lift it."""
        self._last_pick_target = target_name
        print(f"  [PICK] Grasping object at ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})...")
        
        # 0. Open gripper
        self.gripper_ctrl = 1.0
        for _ in range(10):
            action = np.zeros(6, dtype=np.float32)
            self._step(action)

        # 1. HOVER above the object
        hover_target = np.copy(target_pos)
        hover_target[2] += 0.15
        print(f"  [PICK] Phase 1: Hovering above at Z={hover_target[2]:.2f}...")
        
        for _ in range(self.MAX_STEPS):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, hover_target, kp=8.0)
            self._step(action)
            if np.linalg.norm(hover_target - ee_pos) < 0.02:
                break
                
        # 2. LOWER to grasp height
        grasp_target = np.copy(target_pos)
        grasp_target[2] += 0.02
        print(f"  [PICK] Phase 2: Lowering to Z={grasp_target[2]:.2f}...")
        
        for _ in range(self.MAX_STEPS):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, grasp_target, kp=8.0)
            self._step(action)
            if np.linalg.norm(grasp_target - ee_pos) < 0.02:
                break
                
        # 3. CLOSE GRIPPER + ATTACH via constraint
        print(f"  [PICK] Phase 3: Closing gripper and attaching object...")
        self.gripper_ctrl = -1.0
        grasp_hold_pos = np.array(self.env.robot.get_ee_position())
        for _ in range(15):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, grasp_hold_pos, kp=10.0)
            self._step(action)
        
        # Attach the object to the gripper with a fixed constraint
        self._attach_object(target_name)
            
        # 4. LIFT
        lift_target = np.copy(grasp_hold_pos)
        lift_target[2] += 0.20
        print(f"  [PICK] Phase 4: Lifting to Z={lift_target[2]:.2f}...")
        
        for _ in range(self.MAX_STEPS):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, lift_target, kp=5.0)
            self._step(action)
            
            if ee_pos[2] > lift_target[2] - 0.03:
                obj_pos_now = self.get_object_position(target_name)
                if obj_pos_now[2] > target_pos[2] + 0.05:
                    print(f"  [PICK] Object lifted successfully! (obj Z={obj_pos_now[2]:.2f})")
                else:
                    print(f"  [PICK] WARNING: Object may not be grasped (obj Z={obj_pos_now[2]:.2f})")
                break

    def place(self, target_pos: np.ndarray):
        """Hover above target → lower → detach object → release."""
        print(f"  [PLACE] Placing object at ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})...")
        
        drop_z = target_pos[2] + 0.15
        
        # 1. HOVER above the target (aggressive kp for speed)
        hover_target = np.array([target_pos[0], target_pos[1], drop_z + 0.10])
        print(f"  [PLACE] Phase 1: Hovering above at Z={hover_target[2]:.2f}...")
        
        for _ in range(200):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, hover_target, kp=10.0)
            self._step(action)
            if np.linalg.norm(hover_target - ee_pos) < 0.05:
                break

        # 2. LOWER to drop height
        lower_target = np.array([target_pos[0], target_pos[1], drop_z])
        print(f"  [PLACE] Phase 2: Lowering to Z={lower_target[2]:.2f}...")
        
        for _ in range(150):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, lower_target, kp=10.0)
            self._step(action)
            if np.linalg.norm(lower_target - ee_pos) < 0.05:
                break
                
        # 3. DETACH + RELEASE
        print(f"  [PLACE] Phase 3: Releasing object...")
        self._detach_object()
        self.gripper_ctrl = 1.0
        for _ in range(10):
            action = np.zeros(6, dtype=np.float32)
            self._step(action)
        
        # 4. RETRACT upward (quick)
        print(f"  [PLACE] Phase 4: Retracting arm...")
        for _ in range(25):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            retract_target = np.array([ee_pos[0], ee_pos[1], 0.6])
            action[2:5] = self._p_control(ee_pos, retract_target, kp=8.0)
            self._step(action)
        
        print(f"  [PLACE] Done!")

    def toggle(self, target_pos: np.ndarray, target_name: str):
        """Navigate arm toward a device, 'press' it, and toggle its visual state."""
        if target_name not in self.DEVICE_COLORS:
            print(f"  [TOGGLE] '{target_name}' is not a toggleable device.")
            return

        # Reach the arm toward the device
        reach_target = np.copy(target_pos)
        reach_target[2] = min(reach_target[2], 0.45)  # don't reach too high
        print(f"  [TOGGLE] Reaching toward {target_name}...")

        for _ in range(200):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, reach_target, kp=8.0)
            self._step(action)
            if np.linalg.norm(reach_target - ee_pos) < 0.08:
                break

        # Toggle the state
        is_on = self._device_states[target_name]
        new_state = not is_on
        self._device_states[target_name] = new_state

        # Change the object's visual color
        state_key = "on" if new_state else "off"
        new_color = self.DEVICE_COLORS[target_name][state_key]
        body_id = self.env.task.sim._bodies_idx[target_name]
        self._physics_client.changeVisualShape(body_id, -1, rgbaColor=new_color)

        status = "ON" if new_state else "OFF"
        print(f"  [TOGGLE] {target_name.upper()} is now {status}!")

        # Retract arm
        for _ in range(25):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            retract = np.array([ee_pos[0], ee_pos[1], 0.5])
            action[2:5] = self._p_control(ee_pos, retract, kp=8.0)
            self._step(action)

        print(f"  [TOGGLE] Done!")

    def clean(self, target_pos: np.ndarray, target_name: str):
        """Lower arm over dirt, sweep back-and-forth, then remove the dirt."""
        print(f"  [CLEAN] Cleaning '{target_name}' at ({target_pos[0]:.2f}, {target_pos[1]:.2f})...")

        # 1. LOWER arm to just above the floor over the dirt
        sweep_z = 0.08
        center = np.array([target_pos[0], target_pos[1], sweep_z])
        print(f"  [CLEAN] Phase 1: Lowering arm to floor level...")

        for _ in range(150):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            action[2:5] = self._p_control(ee_pos, center, kp=10.0)
            self._step(action)
            if np.linalg.norm(center - ee_pos) < 0.05:
                break

        # 2. SWEEP: move the arm back and forth 3 times
        print(f"  [CLEAN] Phase 2: Sweeping...")
        sweep_range = 0.15
        for sweep in range(3):
            # Sweep right
            right = np.array([center[0] + sweep_range, center[1], sweep_z])
            for _ in range(40):
                ee_pos = self.env.robot.get_ee_position()
                action = np.zeros(6, dtype=np.float32)
                action[2:5] = self._p_control(ee_pos, right, kp=12.0)
                self._step(action)
                if np.linalg.norm(right - ee_pos) < 0.04:
                    break
            # Sweep left
            left = np.array([center[0] - sweep_range, center[1], sweep_z])
            for _ in range(40):
                ee_pos = self.env.robot.get_ee_position()
                action = np.zeros(6, dtype=np.float32)
                action[2:5] = self._p_control(ee_pos, left, kp=12.0)
                self._step(action)
                if np.linalg.norm(left - ee_pos) < 0.04:
                    break

        # 3. REMOVE the dirt from the simulation (teleport it underground)
        print(f"  [CLEAN] Phase 3: Removing dirt...")
        body_id = self.env.task.sim._bodies_idx[target_name]
        self._physics_client.resetBasePositionAndOrientation(
            body_id, [0, 0, -5.0], [0, 0, 0, 1]
        )

        # 4. RETRACT arm
        print(f"  [CLEAN] Phase 4: Retracting arm...")
        for _ in range(25):
            ee_pos = self.env.robot.get_ee_position()
            action = np.zeros(6, dtype=np.float32)
            retract = np.array([ee_pos[0], ee_pos[1], 0.5])
            action[2:5] = self._p_control(ee_pos, retract, kp=8.0)
            self._step(action)

        print(f"  [CLEAN] '{target_name}' has been cleaned!")
