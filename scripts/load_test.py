# File: scripts/load_test.py  
from conveyor import ConveyorBelt  
import grpc  
import conveyor_pb2  
import conveyor_pb2_grpc  
import threading  

def simulate_client():  
    conveyor = ConveyorBelt()  
    channel = grpc.insecure_channel('server-service:50051')  
    stub = conveyor_pb2_grpc.ConveyorServiceStub(channel)  

    while True:  
        data = conveyor.generate_data()  
        request = conveyor_pb2.SensorData(**data)  
        stub.GetOptimizedSettings(request)  

if __name__ == '__main__':  
    for _ in range(50):  
        threading.Thread(target=simulate_client).start()  