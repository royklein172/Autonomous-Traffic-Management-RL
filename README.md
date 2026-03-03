# Autonomous Traffic Management (RL-Based)

This project implements an autonomous traffic signal management system using Reinforcement Learning (PPO) and Computer Vision to optimize vehicle flow in simulated environments.

## Engineering Highlights
- **AI Architecture:** Utilized Proximal Policy Optimization (PPO) for real-time traffic signal timing.
- **Computer Vision:** Implemented YOLOv8 for accurate real-time vehicle detection and traffic flow estimation.
- **Simulation:** Integrated with CARLA/SUMO to model high-fidelity traffic scenarios.
- **Impact:** Achieved a ~30% reduction in average vehicle waiting time in simulated intersection nodes.

## Tech Stack
- **Languages:** Python
- **AI/ML:** PyTorch, Stable-Baselines3, YOLOv8
- **Simulation:** CARLA, SUMO
- **Tools:** OpenCV, NumPy

## How to Run
1. Install dependencies: `pip install -r requirements.txt` (if applicable)
2. Run the training pipeline: `python train.py`
3. Evaluate the agent: `test_sumo.py`

## Role
Developed as a core team project. My focus was on the RL agent training pipeline and integrating the vision system with the traffic signal controller.