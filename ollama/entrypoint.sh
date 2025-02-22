#!/bin/sh

MODEL_NAME="tinyllama"

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready
until curl -s http://localhost:11434/api/version > /dev/null; do
  echo "Waiting for Ollama to start..."
  sleep 5
done

# Pull the model if it's not already present
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "Downloading model $MODEL_NAME..."
  ollama pull "$MODEL_NAME"
fi

# Keep Ollama running in the foreground
wait