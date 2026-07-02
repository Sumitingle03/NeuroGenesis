"""NeuroGenesis — Autonomous Smart Robot Agent.

This script boots up the PyBullet physics simulator and runs the robot
in autonomous mode.  On startup the robot performs housekeeping chores
(cleaning dirt, disposing of trash).  Afterwards it continuously monitors
the environment for displaced objects and picks them up.  The user can
also type commands at any time via the terminal.
"""

import sys
import os
import time
import threading
import queue
import difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.custom_env import CustomHouseholdPandaEnv
from llm_agent.robot_skills import RobotSkills
from llm_agent.brain import LLMBrain

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────
SYNONYMS = {
    "table": "dining_table",
    "desk": "dining_table",
    "counter": "kitchen_counter",
    "kitchen": "kitchen_counter",
    "light": "lamp",
    "bulb": "lamp",
    "television": "tv",
    "screen": "tv",
    "bin": "dustbin",
    "trash_can": "dustbin",
    "garbage": "trash",
    "dirt": "dirt",
    "spill": "dirt",
    "mess": "dirt",
    "cereal": "cereal_box",
    "box": "cereal_box",
}

AVAILABLE_OBJECTS = [
    # Furniture (destinations)
    "kitchen_counter", "dining_table", "shelf", "dustbin",
    # Interactable objects
    "cereal_box", "book", "trash",
    # Smart devices
    "tv", "lamp",
    # Cleanable
    "dirt",
]

# Objects that should NOT be on the floor — if detected on the floor the
# robot will autonomously pick them up and place them on the dining table.
GRASPABLE_OBJECTS = ["cereal_box", "book"]

# Floor threshold — any graspable object whose Z < this value is "on the floor"
FLOOR_Z_THRESHOLD = 0.08


# ────────────────────────────────────────────────────────────
# Helper: Background input thread
# ────────────────────────────────────────────────────────────
def input_thread(cmd_queue: queue.Queue):
    """Runs in a daemon thread; reads user input without blocking physics."""
    while True:
        try:
            cmd = input().strip()
            if cmd:
                cmd_queue.put(cmd)
                if cmd.lower() in ("exit", "quit"):
                    break
        except EOFError:
            break


# ────────────────────────────────────────────────────────────
# Helper: Resolve NLP target names
# ────────────────────────────────────────────────────────────
def resolve_target(target_name: str) -> str:
    """Map synonyms and fuzzy-match the target to a known object name."""
    if target_name in SYNONYMS:
        return SYNONYMS[target_name]

    matches = difflib.get_close_matches(target_name, AVAILABLE_OBJECTS, n=1, cutoff=0.4)
    if matches:
        return matches[0]

    for obj in AVAILABLE_OBJECTS:
        if target_name in obj or obj in target_name:
            return obj

    return target_name  # return as-is; will fail gracefully later


# ────────────────────────────────────────────────────────────
# Helper: Execute a single command (LLM plan → skill calls)
# ────────────────────────────────────────────────────────────
def execute_command(brain, skills, command: str, label: str = "USER") -> bool:
    """Parse a natural-language command and execute the resulting plan.
    
    Returns True on success, False on failure.
    """
    print(f"\n[{label}]: {command}")
    plan = brain.parse_command(command, AVAILABLE_OBJECTS)

    if not plan:
        print("[!] The brain failed to generate a valid plan. Try rephrasing.")
        return False

    print(f"\n[*] Executing {len(plan)} step(s)...")
    for i, step in enumerate(plan):
        skill_name = step.get("skill")
        target_name = resolve_target(step.get("target", ""))

        print(f"\n--- Step {i+1}/{len(plan)}: {skill_name}({target_name}) ---")

        try:
            target_pos = skills.get_object_position(target_name)
        except ValueError as e:
            print(f"[!] Error: {e}")
            return False

        if skill_name == "navigate_to":
            skills.navigate_to(target_pos, target_name=target_name)
        elif skill_name == "pick":
            skills.pick(target_pos, target_name=target_name)
        elif skill_name == "place":
            skills.place(target_pos)
        elif skill_name == "toggle":
            skills.toggle(target_pos, target_name=target_name)
        elif skill_name == "clean":
            skills.clean(target_pos, target_name=target_name)
        else:
            print(f"[!] Unknown skill: {skill_name}")
            return False

    print("[*] Sequence complete!")
    return True


# ────────────────────────────────────────────────────────────
# Helper: Scan for objects displaced onto the floor
# ────────────────────────────────────────────────────────────
def scan_for_displaced_objects(skills, blacklist: set) -> str | None:
    """Return the name of the first graspable object found on the floor, or None.
    
    Objects that have fallen through the world (Z < -1) are added to the
    blacklist so the robot stops chasing them forever.
    """
    for obj_name in GRASPABLE_OBJECTS:
        if obj_name in blacklist:
            continue
        try:
            pos = skills.get_object_position(obj_name)
            # Object has clipped through the world — give up on it
            if pos[2] < -1.0:
                print(f"  [!] '{obj_name}' has fallen out of the world (Z={pos[2]:.1f}). Blacklisting.")
                blacklist.add(obj_name)
                continue
            if pos[2] < FLOOR_Z_THRESHOLD:
                return obj_name
        except ValueError:
            continue  # object may have been removed (e.g. trash after disposal)
    return None


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NeuroGenesis — Autonomous Smart Robot")
    print("=" * 60)

    # 1. Boot environment
    print("[*] Booting up simulation environment...")
    env = CustomHouseholdPandaEnv(render_mode="human")
    obs, info = env.reset()
    print("[*] Environment ready!")

    # 2. Skill, Brain & Camera layers
    from llm_agent.camera_system import CameraSystem
    camera = CameraSystem(env)
    skills = RobotSkills(env, camera_system=camera)
    brain  = LLMBrain(model_name="llama3")

    # 3. Background input thread
    cmd_queue = queue.Queue()
    threading.Thread(target=input_thread, args=(cmd_queue,), daemon=True).start()

    # 4. Autonomous startup chores
    autonomous_chores = [
        "clean the dirt on the floor",
        "put the trash in the dustbin",
    ]

    print("\n" + "-" * 60)
    print("System ready! The robot will begin autonomous chores in 5 seconds...")
    print("You can type a command at any time to override.")
    print("Type 'exit' or 'quit' to close.")
    print("-" * 60)

    # ── 10-second startup delay (keeps physics alive) ──
    lost_objects = set()  # objects that fell through the world — stop chasing them
    delay_start = time.time()
    while time.time() - delay_start < 5:
        # If user types something during the delay, process it immediately
        if not cmd_queue.empty():
            command = cmd_queue.get()
            if command.lower() in ("exit", "quit"):
                print("\nShutting down NeuroGenesis...")
                env.close()
                return
            execute_command(brain, skills, command, label="USER COMMAND")
        camera.render_dashboard()
        env.sim.step()
        time.sleep(0.01)

    print("\n[AUTONOMOUS SYSTEM]: Startup delay complete. Beginning chores...")

    # ── Main loop ──
    pickup_attempts = {}  # track how many times we've tried to pick up each object
    MAX_PICKUP_ATTEMPTS = 2  # give up after this many tries
    
    try:
        while True:
            # Priority 1 — User commands
            if not cmd_queue.empty():
                command = cmd_queue.get()
                if command.lower() in ("exit", "quit"):
                    break
                execute_command(brain, skills, command, label="USER COMMAND")

            # Priority 2 — Startup chores
            elif len(autonomous_chores) > 0:
                chore = autonomous_chores.pop(0)
                print(f"\n{'='*50}")
                print(f"[AUTONOMOUS SYSTEM]: Initiating chore → '{chore}'")
                print(f"{'='*50}")
                execute_command(brain, skills, chore, label="AUTONOMOUS")

                if len(autonomous_chores) == 0:
                    print("\n[AUTONOMOUS SYSTEM]: All startup chores completed!")
                    print("[AUTONOMOUS SYSTEM]: Now monitoring for displaced objects...")

            # Priority 3 — Floor monitoring (detect displaced objects)
            elif (displaced := scan_for_displaced_objects(skills, lost_objects)) is not None:
                # Check retry limit
                pickup_attempts[displaced] = pickup_attempts.get(displaced, 0) + 1
                if pickup_attempts[displaced] > MAX_PICKUP_ATTEMPTS:
                    print(f"\n[AUTONOMOUS SYSTEM]: Gave up on '{displaced}' after {MAX_PICKUP_ATTEMPTS} attempts. Blacklisting.")
                    lost_objects.add(displaced)
                    continue
                    
                print(f"\n{'='*50}")
                print(f"[AUTONOMOUS SYSTEM]: Detected '{displaced}' on the floor! (attempt {pickup_attempts[displaced]}/{MAX_PICKUP_ATTEMPTS})")
                print(f"{'='*50}")
                cmd = f"pick up the {displaced} and put it on the dining table"
                success = execute_command(brain, skills, cmd, label="AUTONOMOUS")
                
                # If successfully placed, check if it's actually on the table now
                if success:
                    time.sleep(1)  # let physics settle
                    try:
                        pos = skills.get_object_position(displaced)
                        if pos[2] > FLOOR_Z_THRESHOLD:
                            # Object is off the floor — reset its counter
                            pickup_attempts[displaced] = 0
                    except ValueError:
                        pass

            # Priority 4 — Idle (keep physics alive)
            else:
                camera.render_dashboard()
                env.sim.step()
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Gracefully handle PyBullet disconnect (user closed the window)
        if "Not connected" in str(e):
            print("\n[!] Simulation window was closed.")
        else:
            print(f"\n[!] Unexpected error: {e}")

    print("\nShutting down NeuroGenesis...")
    import cv2
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
