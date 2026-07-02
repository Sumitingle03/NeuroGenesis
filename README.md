# NeuroGenesis

NeuroGenesis is a simulated robotic agent that uses LLMs and reinforcement learning to autonomously perform household tasks in PyBullet. It can pick up misplaced objects, clean up spills, and follow natural language commands.

We trained a PyBullet Panda robot using Proximal Policy Optimization (PPO) for the low-level pick-and-place skills, and we use an LLM for the high-level reasoning and command parsing.

## Setup

It's recommended to use a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Running the Agent

To start the main autonomous robot:
```bash
python neurogenesis_smart_robot.py
```
You can type commands into the terminal while it runs (like "clean the kitchen" or "put the cereal box on the table").

## Training

If you want to train the RL policies yourself (using stable-baselines3):
```bash
# Train pick and place
python training/train_ppo_pickplace.py
```
