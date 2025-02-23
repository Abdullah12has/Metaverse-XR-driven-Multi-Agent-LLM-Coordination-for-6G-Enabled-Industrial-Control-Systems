import random
import json
from conveyor import ConveyorBelt

def generate_optimization_data(num_samples):
    data = []
    for _ in range(num_samples):
        conveyor = ConveyorBelt()
        conveyor.generate_data()

        # Simulate optimal settings
        optimal_speed = conveyor.speed * 1.05
        optimal_voltage = conveyor.voltage * 0.95

        # Append data
        data.append({
            "input": f"Sensor data: speed={conveyor.speed}, voltage={conveyor.voltage}, load={conveyor.load}, current={conveyor.current}, power={conveyor.power}, rpm={conveyor.rpm}, motor_torque={conveyor.motor_torque}, efficiency={conveyor.efficiency}.",
            "output": f"Optimal settings: new_speed={optimal_speed}, new_voltage={optimal_voltage}."
        })
    return data

# Generate optimization data
optimization_data = generate_optimization_data(10000)

# Save to JSON file
with open("optimization_data.json", "w") as f:
    json.dump(optimization_data, f, indent=4)

print("Optimization data saved to optimization_data.json")