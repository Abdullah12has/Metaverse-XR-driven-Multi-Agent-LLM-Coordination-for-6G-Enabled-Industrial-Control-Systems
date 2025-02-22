import grpc
import conveyor_pb2
import conveyor_pb2_grpc
import time
from client.conveyor import ConveyorBelt

def run():
    # Initialize conveyor system and gRPC client
    conveyor = ConveyorBelt()
    channel = grpc.insecure_channel('localhost:50051')
    stub = conveyor_pb2_grpc.ConveyorServiceStub(channel)

    while True:
        # Generate sensor data from the conveyor system
        sensor_data = conveyor.generate_data()

        # Create a request to send the sensor data to the server
        request = conveyor_pb2.SensorData(
            time=sensor_data["time"],
            speed=sensor_data["speed"],
            voltage=sensor_data["voltage"],
            load=sensor_data["load"],
            current=sensor_data["current"],
            power=sensor_data["power"],
            rpm=sensor_data["rpm"],
            motor_torque=sensor_data["motor_torque"],
            efficiency=sensor_data["efficiency"]
        )

        # Send the sensor data to the server and get optimized settings
        response = stub.GetOptimizedSettings(request)
        print(f"Optimized settings received: {response}")

        # You can compare efficiency before and after optimization here
        print(f"Efficiency before optimization: {sensor_data['efficiency']}")
        print(f"Efficiency after optimization: {sensor_data['efficiency']}")

        time.sleep(1)  # Generate data every second

if __name__ == '__main__':
    run()
