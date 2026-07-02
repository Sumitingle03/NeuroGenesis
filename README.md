# NeuroGenesis

NeuroGenesis is a simulated robotic agent that uses LLMs and reinforcement learning to autonomously perform household tasks in PyBullet. It can pick up misplaced objects, clean up spills, and follow natural language commands.

We trained a PyBullet Panda robot using Proximal Policy Optimization (PPO) for the low-level pick-and-place skills, and we use an LLM for the high-level reasoning and command parsing.

## Setup

It's recommended to use a virtual environment:
```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.venv\Scripts\activate.bat
# Linux/Mac
source .venv/bin/activate
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

## Implimentation

Env
<img width="1462" height="1076" alt="neurogenesis_image" src="https://github.com/user-attachments/assets/d8ad933f-e96f-4f49-868a-9e8afcad42c2" />

# Tasks->

-Dirt cleaning

https://github.com/user-attachments/assets/db13d28b-ca78-4f03-860c-f0f07543fe0f

-Trash Cleaning


https://github.com/user-attachments/assets/893abe53-ca3b-40f3-8ed4-7b7176ae276e

-Object Picking


https://github.com/user-attachments/assets/e651d0e3-cb9a-414c-9be3-7b63af5b2786





