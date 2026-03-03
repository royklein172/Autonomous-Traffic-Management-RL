import traci

def evaluate_static_light(net_file, rou_file, duration=5000): # שינינו ל-5000
    traci.start(["sumo", "-n", net_file, "-r", rou_file])
    
    tl_id = traci.trafficlight.getIDList()[0]
    total_waiting_time = 0
    
    for step in range(duration):
        traci.simulationStep()
        
        # החלפת פאזה כל 30 שניות (300 צעדים)
        if step % 300 == 0:
            current_phase = traci.trafficlight.getPhase(tl_id)
            traci.trafficlight.setPhase(tl_id, (current_phase + 1) % 2)
        
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        total_waiting_time += sum([traci.lane.getWaitingTime(l) for l in lanes])
        
    traci.close()
    
    # המדד להשוואה: זמן המתנה ממוצע לצעד
    avg_waiting_per_step = total_waiting_time / duration
    return total_waiting_time, avg_waiting_per_step

# הגדרת נתיבים
NET_PATH = "/Users/royklein/Desktop/Final_Proj/t_junction.net.xml"
ROU_PATH = "/Users/royklein/Desktop/Final_Proj/t_junction.rou.xml"

print("--- Running Baseline (Static Light - 5000 steps) ---")
total_wt, avg_wt_step = evaluate_static_light(NET_PATH, ROU_PATH)

print(f"\n--- Results ---")
print(f"Total Waiting Time: {total_wt:.2f}")
print(f"Waiting Time per Simulation Step: {avg_wt_step:.2f}")