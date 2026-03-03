import gymnasium as gym
import numpy as np

class BaseTrafficEnv(gym.Env):
    def __init__(self, num_lanes=3):
        """
        Base Environment for Traffic Signal Control.
        This class defines the shared logic for both SUMO and YOLO implementations.
        """
        super(BaseTrafficEnv, self).__init__()
        
        self.num_lanes = num_lanes
        
        # Action Space: 0 = Stay in current phase, 1 = Switch to next green phase
        self.action_space = gym.spaces.Discrete(2)
        
        # Observation Space: [Count, Dist, WaitTime] per lane
        # Total input size = num_lanes * 3. Normalized to [0, 1].
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_lanes * 3,), dtype=np.float32
        )
        
        # Internal state shared by all subclasses (SUMO or YOLO)
        self.lane_data = {
            f'lane_{i}': {'count': 0, 'dist': 1.0, 'wait_time': 0} 
            for i in range(num_lanes)
        }

    def _get_obs(self):
        """
        Converts internal lane_data into a normalized numpy array for the model.
        Normalization ensures the Neural Network processes data in a stable range.
        """
        obs = []
        for i in range(self.num_lanes):
            data = self.lane_data[f'lane_{i}']
            
            # 1. Vehicle Count: Normalized by 10 (assumes max capacity of 10 cars)
            obs.append(np.clip(data['count'] / 10.0, 0, 1))      
            
            # 2. Distance: Already normalized 0.0 (near) to 1.0 (far)
            obs.append(np.clip(data['dist'], 0, 1))             
            
            # 3. Wait Time: Normalized by 100 (frustration threshold)
            obs.append(np.clip(data['wait_time'] / 100.0, 0, 1)) 
            
        return np.array(obs, dtype=np.float32)

    def compute_reward(self, action):
        """
        Calculates the reward (score) to guide the agent.
        The formula balances throughput, average wait time, and fair treatment of all lanes.
        """
        lane_waits = [l['wait_time'] for l in self.lane_data.values()]
        
        total_vehicles = sum([l['count'] for l in self.lane_data.values()])
        total_wait = sum(lane_waits)
        
        # Max Wait: The longest time a single vehicle has been waiting.
        # This prevents 'lane starvation' (where the model ignores a low-traffic lane).
        max_wait = max(lane_waits) if lane_waits else 0
        
        # Reward Formula:
        # Penalizes total congestion, total waiting, and especially the longest waiting driver.
        # Reward = -(TotalVehicles * 0.4) - (TotalWait * 0.8) - (MaxWait * 10.0)
        reward = -(total_vehicles * 0.4) - (total_wait * 0.8) - (max_wait * 10.0)
        
        # Switch Penalty: A small cost to prevent rapid, unnecessary phase flickering.
        if action == 1:
            reward -= 0.2  
            
        return reward

    def reset(self, seed=None, options=None):
        """
        Resets the internal environment state to defaults.
        """
        for i in range(self.num_lanes):
            self.lane_data[f'lane_{i}'] = {'count': 0, 'dist': 1.0, 'wait_time': 0}
        return self._get_obs(), {}