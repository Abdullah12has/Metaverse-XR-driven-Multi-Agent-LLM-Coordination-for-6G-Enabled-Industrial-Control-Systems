import random
import json
from conveyor import ConveyorBelt

def generate_safety_data(num_samples):
    data = []
    for _ in range(num_samples):
        conveyor = ConveyorBelt()
        conveyor.generate_data()

        # Simulate unsafe conditions
        if conveyor.load > 40:  # Overload condition
            corrective_action = "reduce_speed"
        elif conveyor.current > 3.0:  # Overheating condition
            corrective_action = "stop_conveyor"
        else:
            corrective_action = "none"

        # Append data
        data.append({
            "input": f"Sensor data: speed={conveyor.speed}, voltage={conveyor.voltage}, load={conveyor.load}, current={conveyor.current}, power={conveyor.power}, rpm={conveyor.rpm}, motor_torque={conveyor.motor_torque}, efficiency={conveyor.efficiency}.",
            "output": f"Corrective action: {corrective_action}."
        })
    return data

# Generate safety data
safety_data = generate_safety_data(10000)

# Save to JSON file
with open("safety_data.json", "w") as f:
    json.dump(safety_data, f, indent=4)

print("Safety data saved to safety_data.json")