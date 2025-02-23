# Use an official Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install dependencies, including influxdb-client
RUN pip install --no-cache-dir crewai crewai-tools requests flask flask-cors flask-restful influxdb-client

# Set environment variable to point to Ollama service
ENV OLLAMA_API_BASE="http://ollama:11434"

# Run the application
CMD ["python", "api.py"]
