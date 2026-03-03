import gymnasium as gym
import numpy as np
import traci
import os

class TrafficSignalEnv(gym.Env):
    def __init__(self, net_file, rou_file, use_gui=False):
        super(TrafficSignalEnv, self).__init__()
        self.net_file = net_file
        self.rou_file = rou_file
        self.use_gui = use_gui
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # סגירה בטוחה של כל סימולציה קודמת שנתקעה
        try:
            traci.close()
        except:
            pass
            
        # בחירת מצב ריצה: sumo או sumo-gui
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([sumo_binary, "-n", self.net_file, "-r", self.rou_file])
        
        self.tl_id = traci.trafficlight.getIDList()[0]
        
        # החזרה למצב התחלתי נקי
        return np.zeros(4, dtype=np.float32), {}
    
    def step(self, action):
        # 1. בצע את הפעולה
        if action == 1:
            current_phase = traci.trafficlight.getPhase(self.tl_id)
            traci.trafficlight.setPhase(self.tl_id, (current_phase + 1) % 2)
        
        # 2. צעד בסימולציה
        traci.simulationStep()
        
        # 3. איסוף נתונים לכל נתיב
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        # איסוף נתונים למדדים
        waiting_times = [traci.lane.getWaitingTime(l) for l in lanes]
        total_waiting_time = sum(waiting_times)
        halting_numbers = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
        total_halting = sum(halting_numbers)
        avg_speed = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes] or [0])
        phase = traci.trafficlight.getPhase(self.tl_id)
        
        # 4. Observation: נרמול חכם יותר
        # נשתמש בערכי Max גבוהים יותר כדי שהמודל לא יאבד מידע מהר מדי
        obs = np.array([
            np.clip(total_waiting_time/500.0, 0, 1), # נרמול ל-500 שניות
            np.clip(avg_speed/14.0, 0, 1),           # מהירות עד 14 מטר לשנייה (50 קמ"ש)
            phase,                                  # כבר 0 או 1, אין צורך בנרמול
            np.clip(total_halting/50.0, 0, 1)       # עד 50 רכבים תקועים
        ], dtype=np.float32)
        
        # 5. Reward: מבוסס זמן המתנה מצטבר
        # הוספנו עונש קטן על שינוי פאזה (כדי שהרמזור לא יזפזף כמו מטורף)
        reward = -(total_waiting_time / 100.0) 
        if action == 1:
            reward -= 2.0 
        
        terminated = (traci.simulation.getMinExpectedNumber() == 0)
            
        return obs, reward, terminated, False, {}