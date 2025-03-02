#!/bin/bash

# Set environment variables if not already set
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-"8001"}

# Check if running in RunPod mode
if [ -n "$RUNPOD_SERVERLESS" ]; then
    echo "Starting in RunPod serverless mode..."
    python3 -u handler.py
else
    echo "Starting in API server mode..."
    python3 -m uvicorn main:app --host $API_HOST --port $API_PORT
fi