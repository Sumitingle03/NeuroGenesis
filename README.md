# NeuroGenesis — Autonomous Smart Robot Agent

NeuroGenesis is an autonomous smart robot agent that operates in a physics-simulated environment (PyBullet). The robot is capable of performing household chores such as cleaning dirt, disposing of trash, and continuously monitoring the environment for displaced objects (e.g., picking up items from the floor). The project integrates Large Language Model (LLM) reasoning for processing NLP commands with Reinforcement Learning (PPO) for low-level robotic skills like pick-and-place.

## Features
- **Autonomous Mode:** On startup, the robot performs housekeeping chores and continuously monitors the environment for tasks.
- **Natural Language Commands:** Users can type commands in the terminal (via background threads) for the robot to execute asynchronously.
- **Reinforcement Learning:** Includes scripts to train customized pick-and-place policies using Proximal Policy Optimization (PPO) and PyBullet.
- **Robust NLP Matching:** Synonyms and fuzzy-matching logic to map natural language targets to simulated environment objects.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd NeuroGenesis
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the autonomous robot agent:
```bash
python neurogenesis_smart_robot.py
```
While running, you can type commands in the terminal to instruct the robot directly.

To train the pick-and-place reinforcement learning policy:
```bash
python training/train_ppo_pickplace.py
```

## Repository Structure
- `neurogenesis_smart_robot.py`: Main entry point for the smart robot agent.
- `env/`: Custom PyBullet environment configurations.
- `llm_agent/`: Logic for the LLM reasoning (brain), robotic skills mapping, and camera system.
- `training/`: Training scripts using stable-baselines3.
- `requirements.txt`: Python package dependencies.
- `logs/` & `checkpoints/`: (Generated) TensorBoard logs and model checkpoints from RL training.
