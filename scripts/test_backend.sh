#!/bin/bash

# Backend API Test Script for Optionix

# Set up robust error handling
set -euo pipefail

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Determine the project root (one level up from the script)
PROJECT_ROOT="$(dirname "$0")/.."
BACKEND_DIR="$PROJECT_ROOT/code/backend"
VENV_DIR="$PROJECT_ROOT/venv"
BACKEND_PORT=8000
BACKEND_URL="http://localhost:$BACKEND_PORT"

echo -e "${BLUE}Starting backend server for testing...${NC}"

# Check for venv
if [ ! -d "$VENV_DIR" ]; then
  echo -e "${RED}Error: Python virtual environment not found at $VENV_DIR.${NC}"
  echo -e "${RED}Please run the setup script first: ./scripts/setup_optionix_env.sh${NC}"
  exit 1
fi

# Activate venv and start server
cd "$BACKEND_DIR"
source "$VENV_DIR/bin/activate"

# Start the backend server in the background
# Use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!
echo -e "${GREEN}Backend started with PID: ${BACKEND_PID}${NC}"

# Wait for backend to start
echo -e "${BLUE}Waiting for backend to initialize...${NC}"
sleep 5

# --- Test backend endpoints ---
echo -e "\n${BLUE}Testing backend endpoints...${NC}"
TEST_FAILURES=0

# 1. Testing root endpoint
echo -e "${BLUE}1. Testing root endpoint ($BACKEND_URL/)...${NC}"
RESPONSE=$(curl -s "$BACKEND_URL/")
if echo "$RESPONSE" | grep -q "Welcome to Optionix API"; then
  echo -e "${GREEN}PASS: Root endpoint returned expected welcome message.${NC}"
else
  echo -e "${RED}FAIL: Root endpoint test failed. Response: $RESPONSE${NC}"
  TEST_FAILURES=$((TEST_FAILURES + 1))
fi

# 2. Testing volatility prediction endpoint
echo -e "${BLUE}2. Testing volatility prediction endpoint ($BACKEND_URL/predict_volatility)...${NC}"
PREDICTION_DATA='{"open": 42500, "high": 43000, "low": 42000, "volume": 1000000}'
RESPONSE=$(curl -s -X POST "$BACKEND_URL/predict_volatility" \
  -H "Content-Type: application/json" \
  -d "$PREDICTION_DATA")

# Simple check for a valid JSON response structure (e.g., contains "prediction")
if echo "$RESPONSE" | grep -q "prediction"; then
  echo -e "${GREEN}PASS: Volatility prediction endpoint returned a prediction.${NC}"
  echo "Response: $RESPONSE"
else
  echo -e "${RED}FAIL: Volatility prediction endpoint test failed. Response: $RESPONSE${NC}"
  TEST_FAILURES=$((TEST_FAILURES + 1))
fi

# --- Cleanup ---
echo -e "\n${BLUE}Stopping backend server (PID: $BACKEND_PID)...${NC}"
kill "$BACKEND_PID"

# Deactivate venv
deactivate

# --- Final Result ---
if [ "$TEST_FAILURES" -eq 0 ]; then
  echo -e "\n${GREEN}Backend API tests completed successfully! (0 failures)${NC}"
else
  echo -e "\n${RED}Backend API tests completed with $TEST_FAILURES failure(s).${NC}"
  exit 1
fi
