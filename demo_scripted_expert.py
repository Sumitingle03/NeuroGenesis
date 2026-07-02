"""Scripted geometric expert policy for the Pick-and-Place task.

This script uses a simple state machine and proportional control to 
demonstrate a perfect pick-and-place sequence without any RL training.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.custom_env import CustomHouseholdPandaEnv

def main():
    print("=" * 60)
    print("  NeuroGenesis — Scripted Expert Demonstration")
    print("=" * 60)

    env = CustomHouseholdPandaEnv(render_mode="human")
    obs, info = env.reset()

    # State machine phases
    PHASE_APPROACH_BASE = 0
    PHASE_OPEN_GRIPPER = 1
    PHASE_LOWER_ARM = 2
    PHASE_CLOSE_GRIPPER = 3
    PHASE_LIFT_ARM = 4
    PHASE_CARRY_TO_GOAL = 5
    PHASE_LOWER_TO_GOAL = 6
    PHASE_RELEASE = 7
    PHASE_DONE = 8

    phase = PHASE_APPROACH_BASE

    for step_idx in range(500):
        # Read true state from the environment
        # achieved_goal = object position
        # desired_goal = target placement position
        obj_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]
        
        # Get EEF and base positions
        ee_pos = env.robot.get_ee_position()
        
        # In MobilePanda, base_xy is prepended to robot_obs
        base_xy = env.robot.get_obs()[:2]
        
        # Default action: stay still, keep gripper state from previous
        action = np.zeros(6, dtype=np.float32)
        
        # We need to maintain the gripper state across steps
        # Default to fully open initially
        gripper_ctrl = 1.0 

        # -------------------------------------------------------------
        # Proportional Control Logic
        # -------------------------------------------------------------
        
        # Helper for P-control
        def p_control(current, target, kp=5.0, max_vel=1.0):
            diff = target - current
            vel = kp * diff
            norm = np.linalg.norm(vel)
            if norm > max_vel:
                vel = vel * (max_vel / norm)
            return vel

        if phase == PHASE_APPROACH_BASE:
            # Drive base to a comfortable reaching distance from the object
            # (e.g., offset slightly on the X axis so arm reaches forward)
            base_target = np.array([obj_pos[0] - 0.4, obj_pos[1]])
            action[:2] = p_control(base_xy, base_target, kp=3.0)
            
            # Keep arm high and neutral
            ee_target = np.array([base_xy[0] + 0.4, base_xy[1], 0.3])
            action[2:5] = p_control(ee_pos, ee_target)
            
            if np.linalg.norm(base_target - base_xy) < 0.05:
                phase = PHASE_OPEN_GRIPPER

        elif phase == PHASE_OPEN_GRIPPER:
            # Just open the gripper
            gripper_ctrl = 1.0
            phase = PHASE_LOWER_ARM

        elif phase == PHASE_LOWER_ARM:
            # Move EEF directly over the object
            ee_target = np.copy(obj_pos)
            ee_target[2] += 0.02 # Slightly above center to grasp top
            action[2:5] = p_control(ee_pos, ee_target, kp=10.0)
            
            if np.linalg.norm(ee_target - ee_pos) < 0.02:
                phase = PHASE_CLOSE_GRIPPER
                
        elif phase == PHASE_CLOSE_GRIPPER:
            # Close gripper and wait a few steps for physics to grip
            gripper_ctrl = -1.0
            action[5] = gripper_ctrl
            # Execute immediately and hold for 10 steps
            for _ in range(10):
                obs, _, _, _, _ = env.step(action)
                time.sleep(0.02)
            phase = PHASE_LIFT_ARM
            
        elif phase == PHASE_LIFT_ARM:
            gripper_ctrl = -1.0
            # Lift the object up
            ee_target = np.copy(obj_pos)
            ee_target[2] += 0.2
            action[2:5] = p_control(ee_pos, ee_target, kp=5.0)
            
            if ee_pos[2] > obj_pos[2] + 0.15:
                phase = PHASE_CARRY_TO_GOAL
                
        elif phase == PHASE_CARRY_TO_GOAL:
            gripper_ctrl = -1.0
            # Drive base towards the goal
            base_target = np.array([goal_pos[0] - 0.4, goal_pos[1]])
            action[:2] = p_control(base_xy, base_target, kp=3.0)
            
            # Keep arm high while driving
            ee_target = np.array([base_target[0] + 0.4, base_target[1], 0.3])
            action[2:5] = p_control(ee_pos, ee_target)
            
            if np.linalg.norm(base_target - base_xy) < 0.05:
                phase = PHASE_LOWER_TO_GOAL
                
        elif phase == PHASE_LOWER_TO_GOAL:
            gripper_ctrl = -1.0
            # Lower arm to goal position
            ee_target = np.copy(goal_pos)
            action[2:5] = p_control(ee_pos, ee_target, kp=5.0)
            
            if np.linalg.norm(ee_target - ee_pos) < 0.02:
                phase = PHASE_RELEASE
                
        elif phase == PHASE_RELEASE:
            gripper_ctrl = 1.0
            action[5] = gripper_ctrl
            for _ in range(10):
                obs, _, _, _, _ = env.step(action)
                time.sleep(0.02)
            phase = PHASE_DONE
            
        elif phase == PHASE_DONE:
            # Retract arm slightly
            ee_target = np.array([base_xy[0] + 0.3, base_xy[1], 0.4])
            action[2:5] = p_control(ee_pos, ee_target)
            gripper_ctrl = 1.0

        # Apply gripper control
        action[5] = gripper_ctrl

        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.04) # SLOW down so we can see it
        
        if phase == PHASE_DONE and info.get('is_success'):
            print("Successfully placed the object at the target!")
            break

    print("Demo completed.")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    main()
