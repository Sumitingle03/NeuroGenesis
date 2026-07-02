import time
import gymnasium as gym
from gymnasium.envs.registration import register
from env.custom_env import CustomHouseholdPandaEnv

# Register the environment
register(
    id="CustomHouseholdPanda-v0",
    entry_point="env.custom_env:CustomHouseholdPandaEnv",
    max_episode_steps=50,
)

def main():
    print("Initializing Custom Household Environment...")
    env = gym.make("CustomHouseholdPanda-v0", render_mode="human")
    
    obs, info = env.reset()
    print("Environment reset successfully.")
    
    # Run a few episodes with random actions
    num_episodes = 5
    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            step_count = 0
            
            while not done and not truncated:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                # Optional: slow down the simulation to see it
                time.sleep(0.1)
                step_count += 1
                
            print(f"Episode {ep + 1} finished after {step_count} steps.")
    except Exception as e:
        print(f"Simulation ended: {e}")

if __name__ == "__main__":
    main()
