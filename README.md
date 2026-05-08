# Reinforcement Learning for Robotic Manipulation with HER

Implementing Hindsight Experience Replay (HER) combined with Soft Actor-Critic (SAC) 
for goal-conditioned robotic manipulation tasks using MuJoCo and Gymnasium Robotics.

## Overview

This project implements HER from scratch in PyTorch, applied to the FetchReach and 
FetchPush environments. HER addresses the sparse reward problem in goal-conditioned RL 
by treating failed episodes as successful ones with different goals.

## Environments

- **FetchReach-v4**: Robotic arm learns to move end-effector to a target position
- **FetchPush-v4**: Robotic arm learns to push a block to a target position

## Architecture

- `agents/` - SAC agent and HER implementation
- `environments/` - Environment wrappers and utilities
- `utils/` - Replay buffer, logging, helper functions
- `scripts/` - Training and evaluation scripts
- `results/` - Training curves and videos

## Requirements

```bash
pip install torch gymnasium gymnasium-robotics mujoco wandb numpy matplotlib tqdm
```

## Training

Coming soon.

## Results

Coming soon.