import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_env import SumoEnv
import traci
import time

# 1. Paths configuration - Absolute paths are safer
NET_PATH = "t_junction.net.xml"
ROU_PATH = "t_junction.rou.xml"
MODEL_PATH = "traffic_signal_ppo_pro.zip"
VEC_PATH = "vec_normalize.pkl"

# 2. Environment Setup
def make_env():
    return SumoEnv(net_file=NET_PATH, rou_file=ROU_PATH, use_gui=True, num_lanes=3)

env = DummyVecEnv([make_env])

# 3. Load Normalization Statistics
try:
    env = VecNormalize.load(VEC_PATH, env)
    env.training = False  # Critical: do not update stats during testing
    env.norm_reward = False 
    print("--- Normalization loaded successfully ---")
except Exception as e:
    print(f"--- Warning: Could not load VecNormalize: {e} ---")

# 4. Load the Trained Model
print("--- Loading trained model ---")
model = PPO.load(MODEL_PATH, env=env)

# 5. Continuous Test Loop
print("--- Starting Test ---")
print("--- TIP: Press the 'Play' button in the SUMO GUI to start traffic ---")

obs = env.reset()
step_count = 0

try:
    # Changed from range(2000) to while True to prevent sudden disconnection errors
    while True:
        # Predict action (deterministic=True for optimal policy)
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, rewards, dones, infos = env.step(action)
        
        # Optional: Print status every 10 steps to monitor the "brain"
        if step_count % 10 == 0:
            env_instance = env.envs[0]
            counts = [lane['count'] for lane in env_instance.lane_data.values()]
            print(f"Step: {step_count} | Cars per Lane: {counts} | Action taken: {action[0]}")
            
        step_count += 1
        
        # If simulation ends (all cars finished), reset automatically
        if dones[0]:
            print("--- Scenario finished, resetting environment ---")
            obs = env.reset()
            
except KeyboardInterrupt:
    print("\n--- Test stopped by user (Ctrl+C) ---")
except Exception as e:
    print(f"\n--- An error occurred: {e} ---")
finally:
    # Close TraCI connection properly
    if traci.isLoaded():
        traci.close()
    print("--- Simulation closed ---")