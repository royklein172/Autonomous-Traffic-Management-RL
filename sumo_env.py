import traci
import numpy as np
from base_env import BaseTrafficEnv

class SumoEnv(BaseTrafficEnv):
    def __init__(self, net_file, rou_file, use_gui=True, num_lanes=3):
        super(SumoEnv, self).__init__(num_lanes=num_lanes)
        self.net_file = net_file
        self.rou_file = rou_file
        self.use_gui = use_gui
        self.tl_id = None 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
            
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([sumo_binary, "-n", self.net_file, "-r", self.rou_file, "--no-warnings"])
        
        self.tl_id = traci.trafficlight.getIDList()[0]

        # SOLUTION C: Disable SUMO's automatic phase switching
        # Setting a massive duration (10,000s) forces the light to wait for our command.
        traci.trafficlight.setPhaseDuration(self.tl_id, 10000)
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Traffic Light Logic: Action 1 triggers a phase jump
        if action == 1:
            current_phase = traci.trafficlight.getPhase(self.tl_id)
            next_phase = 2 if current_phase == 0 else 0
            traci.trafficlight.setPhase(self.tl_id, next_phase)
            
            # SOLUTION C: Re-freeze the new phase duration so it doesn't time out
            traci.trafficlight.setPhaseDuration(self.tl_id, 10000)
        
        # 2. Advance simulation
        traci.simulationStep()
        
        # 3. Data Collection (Unique lanes + Normalized Distance)
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(dict.fromkeys(controlled_lanes)) 
        
        for i in range(min(self.num_lanes, len(unique_lanes))):
            lane_id = unique_lanes[i]
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            
            dist_val = 1.0 
            if len(vehicles) > 0:
                lane_length = traci.lane.getLength(lane_id)
                leader_pos = traci.vehicle.getLanePosition(vehicles[-1])
                dist_val = np.clip((lane_length - leader_pos) / lane_length, 0, 1)

            self.lane_data[f'lane_{i}'] = {
                'count': len(vehicles),
                'dist': dist_val,
                'wait_time': traci.lane.getWaitingTime(lane_id)
            }
            
        obs = self._get_obs()
        reward = self.compute_reward(action)
        terminated = (traci.simulation.getMinExpectedNumber() == 0)
        
        return obs, reward, terminated, False, {}