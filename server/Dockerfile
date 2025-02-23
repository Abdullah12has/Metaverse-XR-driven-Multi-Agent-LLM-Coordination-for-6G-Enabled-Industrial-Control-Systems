# Use Python 3.9 image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy all files from current directory
COPY . .

# Install dependencies and generate protobuf code
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update \
    && apt-get install -y protobuf-compiler \
    && python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. conveyor.proto

# Expose necessary ports
EXPOSE 50051 8000

# Start the server when the container runs
CMD ["python", "server.py"]