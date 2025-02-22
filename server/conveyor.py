import random
import math
import time

class ConveyorBelt:
    def __init__(self):
        self.speed = 0.0  # m/s
        self.voltage = 0.0  # V
        self.load = 0.0  # kg
        self.current = 0.0  # A
        self.power = 0.0  # W
        self.rpm = 0.0  # Rotational speed
        self.motor_torque = 0.0  # N.m
        self.efficiency = 0.85  # Initial efficiency

    def calculate_efficiency(self):
        """Calculate efficiency dynamically based on power and load."""
        # Assume efficiency is a function of load and voltage
        efficiency_factor = (self.load / (self.load + 10))  # Simplified load factor
        voltage_factor = self.voltage / 24  # Max voltage 24V
        self.efficiency = 0.8 + (efficiency_factor * 0.15) + (voltage_factor * 0.05)
        if self.efficiency > 1.0:
            self.efficiency = 1.0  # Max efficiency cap
        elif self.efficiency < 0.5:
            self.efficiency = 0.5  # Min efficiency cap
        return self.efficiency

    def generate_data(self):
        """Simulate conveyor sensor data with realistic calculations."""
        self.load = random.uniform(1, 50)  # Random load weight in kg
        self.voltage = random.uniform(10, 24)  # Voltage applied to motor
        self.speed = self.voltage * 0.1  # Proportional relationship between voltage and speed
        self.rpm = self.speed * 100  # Convert speed to RPM
        self.current = self.voltage / (self.load + 1)  # Simplified model of current
        self.power = self.voltage * self.current  # Power consumption
        self.motor_torque = self.voltage * 0.1  # Simplified model of motor torque

        efficiency = self.calculate_efficiency()

        # Return sensor data as a dictionary
        return {
            "time": time.time(),
            "speed": self.speed,
            "voltage": self.voltage,
            "load": self.load,
            "current": self.current,
            "power": self.power,
            "rpm": self.rpm,
            "motor_torque": self.motor_torque,
            "efficiency": efficiency
        }
