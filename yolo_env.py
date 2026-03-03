import gymnasium as gym
import numpy as np
from base_env import BaseTrafficEnv

class YoloEnv(BaseTrafficEnv):
    def __init__(self, num_lanes=3):
        super(YoloEnv, self).__init__(num_lanes=num_lanes)
        # We need to keep track of which direction is currently GREEN
        # 0 = West-East Green, 2 = South Green (Matches your SUMO logic)
        self.current_phase = 0 

    def update_data_from_yolo(self, detections):
        """
        detections: { 'lane_0': {'count': n, 'dist': d}, ... }
        Values for 'dist' must be 0.0 (near) to 1.0 (far).
        """
        for i in range(self.num_lanes):
            lane_key = f'lane_{i}'
            if lane_key in detections:
                self.lane_data[lane_key]['count'] = detections[lane_key]['count']
                self.lane_data[lane_key]['dist'] = detections[lane_key]['dist']
                
                # REFINED WAIT TIME LOGIC:
                # We only increment wait_time if there are cars AND the light is RED for this lane.
                # Assuming lane_0 and lane_1 are Phase 0, and lane_2 is Phase 2.
                is_green = False
                if self.current_phase == 0 and i in [0, 1]: is_green = True
                if self.current_phase == 2 and i == 2: is_green = True
                
                if detections[lane_key]['count'] > 0 and not is_green:
                    # Increment 'frustration' because they are stuck at red
                    self.lane_data[lane_key]['wait_time'] += 1
                elif is_green or detections[lane_key]['count'] == 0:
                    # Reset when cars are moving or lane is empty
                    self.lane_data[lane_key]['wait_time'] = 0

    def step(self, action):
        """
        In YOLO mode, 'step' executes the logic and updates the light state.
        """
        # 1. If action is 1, we switch the internal phase
        if action == 1:
            self.current_phase = 2 if self.current_phase == 0 else 0
            # Note: Here you would send a command to the physical Arduino/Controller
            print(f"--- HARDWARE COMMAND: SWITCHING TO PHASE {self.current_phase} ---")
        
        # 2. Return the observation and reward based on latest YOLO data
        obs = self._get_obs()
        reward = self.compute_reward(action)
        
        return obs, reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_phase = 0
        return self._get_obs(), {}