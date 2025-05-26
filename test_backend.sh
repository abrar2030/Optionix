#!/bin/bash

# Start the backend server
echo "Starting backend server..."
cd /home/ubuntu/optionix_project/code/backend
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Test backend endpoints
echo "Testing backend endpoints..."
echo "1. Testing root endpoint"
curl -s http://localhost:8000/ | grep "Welcome to Optionix API"

echo "2. Testing volatility prediction endpoint"
curl -s -X POST http://localhost:8000/predict_volatility \
  -H "Content-Type: application/json" \
  -d '{"open": 42500, "high": 43000, "low": 42000, "volume": 1000000}'

# Kill the backend process when done
echo "Stopping backend server..."
kill $BACKEND_PID

echo "Backend API tests completed."
