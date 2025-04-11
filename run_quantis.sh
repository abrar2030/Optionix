#!/bin/bash

# Run script for Quantis project
# This script starts the API, models, and frontend components

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Quantis application...${NC}"

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo -e "${BLUE}Creating Python virtual environment...${NC}"
  python3 -m venv venv
fi

# Start API server
echo -e "${BLUE}Starting API server...${NC}"
cd api
source ../venv/bin/activate
pip install -r requirements.txt > /dev/null
python app.py &
API_PID=$!
cd ..

# Start model service
echo -e "${BLUE}Starting model service...${NC}"
cd models
source ../venv/bin/activate
python model_service.py &
MODEL_PID=$!
cd ..

# Wait for backend services to initialize
echo -e "${BLUE}Waiting for backend services to initialize...${NC}"
sleep 8

# Start frontend
echo -e "${BLUE}Starting frontend...${NC}"
cd frontend
npm install > /dev/null
npm start &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}Quantis application is running!${NC}"
echo -e "${GREEN}API running with PID: ${API_PID}${NC}"
echo -e "${GREEN}Model service running with PID: ${MODEL_PID}${NC}"
echo -e "${GREEN}Frontend running with PID: ${FRONTEND_PID}${NC}"
echo -e "${GREEN}Access the application at: http://localhost:3000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"

# Handle graceful shutdown
function cleanup {
  echo -e "${BLUE}Stopping services...${NC}"
  kill $FRONTEND_PID
  kill $MODEL_PID
  kill $API_PID
  echo -e "${GREEN}All services stopped${NC}"
  exit 0
}

trap cleanup SIGINT

# Keep script running
wait
