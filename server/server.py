import grpc
from kafka import KafkaConsumer
import json
import os
import threading
from concurrent import futures
import conveyor_pb2
import conveyor_pb2_grpc
from conveyor import ConveyorBelt
from raft import RaftNode
from prometheus_client import start_http_server, Counter

# Metrics setup
start_http_server(8000)
REQUESTS = Counter('grpc_requests_total', 'Total gRPC requests')

# Kafka consumer
consumer = KafkaConsumer(
    'sensor_topic',
    bootstrap_servers='kafka.default.svc.cluster.local:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def kafka_consumer():
    for message in consumer:
        data = message.value
        print(f"Kafka received: {data}")

class ConveyorService(conveyor_pb2_grpc.ConveyorServiceServicer):
    def __init__(self):
        self.conveyor = ConveyorBelt()
        self.pod_ip = os.getenv('POD_IP', 'localhost')
        self.raft = RaftNode(self.pod_ip)
        self.raft.start_election()  # Start election on init

    def GetOptimizedSettings(self, request, context):
        REQUESTS.inc()
        if not self.raft.is_leader():
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"Not leader node. Leader is at {self.raft.leader_address}"
            )
        
        optimized = self.optimize_system(request)
        return conveyor_pb2.OptimizedSettings(**optimized)

    def optimize_system(self, sensor_data):
        return {
            'new_speed': sensor_data.speed * 1.05,
            'new_voltage': sensor_data.voltage * 0.95,
            'new_pid_kp': 1.2,
            'new_pid_ki': 0.6,
            'new_pid_kd': 0.15
        }

def serve():
    # Start Kafka consumer thread
    threading.Thread(target=kafka_consumer, daemon=True).start()
    
    # gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    conveyor_pb2_grpc.add_ConveyorServiceServicer_to_server(
        ConveyorService(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()