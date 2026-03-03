from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_env import SumoEnv  # Updated import
import os

# --- Configuration ---
NET_PATH = "/Users/royklein/Desktop/Final_Proj/t_junction.net.xml"
ROU_PATH = "/Users/royklein/Desktop/Final_Proj/t_junction.rou.xml"
NUM_LANES = 3 # Define how many lanes your model will learn to control

print("--- Initializing Modular Environment ---")

# Step 1: Create the environment function
# We use the new SumoEnv class we built together
def make_env():
    return SumoEnv(
        net_file=NET_PATH,
        rou_file=ROU_PATH,
        use_gui=False,   # Keep GUI off for faster training
        num_lanes=NUM_LANES
    )

# Step 2: Wrap in DummyVecEnv
env = DummyVecEnv([make_env])

# Step 3: VecNormalize - Extremely important for Reward Scaling
# Since we already normalize Obs in base_env, VecNormalize will provide 
# an extra layer of stability, especially for the Rewards.
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

print("--- Setting up PPO Model (Architecture: 256x256) ---")
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    n_steps=2048, 
    batch_size=64, 
    learning_rate=0.0003,
    policy_kwargs=dict(net_arch=[256, 256]) # Deeper network for complex T-junction patterns
)

print("--- Starting Training (100,000 Timesteps) ---")
# This might take some time depending on your CPU
model.learn(total_timesteps=100000) 

print("--- Training Finished! ---")

# Step 4: Save everything
model.save("traffic_signal_ppo_pro")
env.save("vec_normalize.pkl") 

print("--- Model and Normalizer Saved Successfully! ---")