#!/bin/bash

# Optionix Project Setup Script (Comprehensive)

# Exit immediately if a command exits with a non-zero status.
set -e

# Prerequisites (ensure these are installed):
# - Python 3.x (the script will use python3.11 available in the environment)
# - pip (Python package installer)
# - Node.js (for frontend)
# - npm (Node package manager)

echo "Starting Optionix project setup..."

PROJECT_DIR="/home/ubuntu/projects_extracted/Optionix"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Error: Project directory ${PROJECT_DIR} not found."
  echo "Please ensure the project is extracted correctly."
  exit 1
fi

cd "${PROJECT_DIR}"
echo "Changed directory to $(pwd)"

# --- Backend Setup (FastAPI/Python) ---
echo ""
echo "Setting up Optionix Backend..."
BACKEND_DIR="${PROJECT_DIR}/code/backend"

if [ ! -d "${BACKEND_DIR}" ]; then
    echo "Error: Backend directory ${BACKEND_DIR} not found. Skipping backend setup."
else
    cd "${BACKEND_DIR}"
    echo "Changed directory to $(pwd) for backend setup."

    if [ ! -f "requirements.txt" ]; then
        echo "Error: requirements.txt not found in ${BACKEND_DIR}. Cannot install backend dependencies."
    else
        echo "Creating Python virtual environment for backend (venv_optionix_backend_py)..."
        if ! python3.11 -m venv venv_optionix_backend_py; then # Using a specific venv name
            echo "Failed to create backend virtual environment. Please check your Python installation."
        else
            source venv_optionix_backend_py/bin/activate
            echo "Backend Python virtual environment created and activated."

            echo "Installing backend Python dependencies from requirements.txt..."
            pip3 install -r requirements.txt
            echo "Backend dependencies installed."

            echo "To activate the backend virtual environment later, run: source ${BACKEND_DIR}/venv_optionix_backend_py/bin/activate"
            echo "To start the backend server (from ${BACKEND_DIR} with venv activated): uvicorn app:app --host 0.0.0.0 --port 8000 (as per README)"
            deactivate
            echo "Backend Python virtual environment deactivated."
        fi
    fi
    cd "${PROJECT_DIR}" # Return to the main project directory
fi

# --- Frontend Setup (React/Node.js) ---
echo ""
echo "Setting up Optionix Web Frontend..."
# The README refers to code/frontend, but the ls output showed code/web-frontend for package.json
# We will use code/web-frontend as it contains the package.json
FRONTEND_DIR="${PROJECT_DIR}/code/web-frontend"

if [ ! -d "${FRONTEND_DIR}" ]; then
    # Fallback to code/frontend if code/web-frontend doesn't exist, as per README structure diagram
    if [ -d "${PROJECT_DIR}/code/frontend" ]; then
        FRONTEND_DIR="${PROJECT_DIR}/code/frontend"
        echo "Note: Using ${FRONTEND_DIR} as web-frontend directory was not found."
    else
        echo "Error: Frontend directory (neither ${PROJECT_DIR}/code/web-frontend nor ${PROJECT_DIR}/code/frontend) not found. Skipping frontend setup."
        FRONTEND_DIR=""
    fi
fi

if [ -n "${FRONTEND_DIR}" ]; then
    cd "${FRONTEND_DIR}"
    echo "Changed directory to $(pwd) for frontend setup."

    if [ ! -f "package.json" ]; then
        echo "Error: package.json not found in ${FRONTEND_DIR}. Cannot install frontend dependencies."
    else
        echo "Installing frontend Node.js dependencies using npm..."
        if ! command -v npm &> /dev/null; then
            echo "npm command could not be found. Please ensure Node.js and npm are installed and in your PATH."
        else
            npm install
            echo "Frontend dependencies installed."
            echo "To start the frontend development server (from ${FRONTEND_DIR}): npm start (or webpack serve --mode development --open as per package.json)"
            echo "To build the frontend for production (from ${FRONTEND_DIR}): npm run build (or webpack --mode production as per package.json)"
        fi
    fi
    cd "${PROJECT_DIR}" # Return to the main project directory
fi

# --- AI Models & Blockchain components (Placeholder based on README structure) ---
echo ""
echo "Notes on other components mentioned in README (AI Models, Blockchain):"
AI_MODELS_DIR_OPTIONIX="${PROJECT_DIR}/code/ai_models"
BLOCKCHAIN_DIR_OPTIONIX="${PROJECT_DIR}/code/blockchain"

if [ -d "${AI_MODELS_DIR_OPTIONIX}" ]; then
    echo "- An 'ai_models' directory exists at ${AI_MODELS_DIR_OPTIONIX}. Check for specific setup instructions or dependency files (e.g., requirements.txt) within it."
    if [ -f "${AI_MODELS_DIR_OPTIONIX}/requirements.txt" ]; then
        echo "  Found requirements.txt in ${AI_MODELS_DIR_OPTIONIX}. Consider setting up a separate Python environment for it."
    fi
else
    echo "- 'ai_models' directory not found at ${AI_MODELS_DIR_OPTIONIX}."
fi

if [ -d "${BLOCKCHAIN_DIR_OPTIONIX}" ]; then
    echo "- A 'blockchain' directory exists at ${BLOCKCHAIN_DIR_OPTIONIX}. Check for specific setup instructions or dependency files (e.g., package.json for Hardhat/Truffle, or Python requirements) within it."
    if [ -f "${BLOCKCHAIN_DIR_OPTIONIX}/package.json" ]; then
        echo "  Found package.json in ${BLOCKCHAIN_DIR_OPTIONIX}. It might be a Node.js based blockchain project (e.g., Hardhat, Truffle)."
    fi
else
    echo "- 'blockchain' directory not found at ${BLOCKCHAIN_DIR_OPTIONIX}."
fi

echo ""
echo "Optionix project setup script finished."
echo "Please ensure all prerequisites (Python, Node.js, npm) are installed."
echo "Review the project's README.md and the instructions above for running the backend and frontend."
echo "Manual setup may be required for AI Models and Blockchain components if specific dependency files were not found or processed by this script."
