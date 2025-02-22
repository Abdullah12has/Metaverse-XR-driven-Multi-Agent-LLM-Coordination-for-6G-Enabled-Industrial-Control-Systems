import grpc
from kafka import KafkaProducer
import json
import time
import conveyor_pb2
import conveyor_pb2_grpc
from conveyor import ConveyorBelt

# Kafka setup
producer = KafkaProducer(
    bootstrap_servers='kafka.default.svc.cluster.local:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
def send_to_kafka(data):
    producer.send('sensor_topic', data)

def run():
    conveyor = ConveyorBelt()
    leader_address = 'server-service:50051'  # Default service address
    retries = 3
    retry_delay = 2  # Seconds

    while True:
        sensor_data = conveyor.generate_data()
        sensor_data['lamport_time'] = time.time_ns()
        send_to_kafka(sensor_data)

        request = conveyor_pb2.SensorData(**sensor_data)
        
        for attempt in range(retries):
            try:
                channel = grpc.insecure_channel(leader_address)
                stub = conveyor_pb2_grpc.ConveyorServiceStub(channel)
                response = stub.GetOptimizedSettings(request)
                print(f"Optimized settings received: {response}")
                break  # Success - exit retry loop
            except grpc.RpcError as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if "Leader is at" in str(e):
                    # Extract leader address from error message
                    new_leader = str(e).split()[-1].strip()
                    if new_leader and ':' in new_leader:
                        leader_address = new_leader
                        print(f"Redirecting to leader: {leader_address}")
                    else:
                        leader_address = 'server-service:50051'  # Reset
                time.sleep(retry_delay)
        else:
            print("All retries failed. Resetting to default leader.")
            leader_address = 'server-service:50051'

        time.sleep(1)

if __name__ == '__main__':
    run()