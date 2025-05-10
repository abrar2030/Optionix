# Optionix Project Documentation

## Overview
This document provides an overview of the Optionix project structure, components, and improvements made.

<div align="center">
  <img src="docs/Optionix.bmp" alt="Trading Derivatives Platform" width="100%">
</div>

> **Note**: This Project is currently under active development. Features and functionalities are being added and improved continuously to enhance user experience.

## Project Structure

```
optionix_project/
├── code/
│   ├── ai_models/            # AI models for volatility prediction
│   ├── backend/              # FastAPI backend server
│   ├── blockchain/           # Blockchain contracts and integration
│   ├── frontend/             # React frontend application
│   └── quantitative/         # Quantitative finance models
├── test_backend.sh           # Script for testing backend endpoints
└── todo.md                   # Project task tracking
```

## Components

### Backend (FastAPI)
- API endpoints for options pricing and volatility prediction
- Integration with blockchain contracts
- Error handling and validation

### Frontend (React)
- Modern UI with responsive design
- Dashboard with market overview and portfolio summary
- Trading interface with option chain and order book
- Portfolio management with position tracking
- Analytics with risk assessment and volatility charts

### AI Models
- Volatility prediction model
- Training scripts for model generation

### Blockchain Integration
- Smart contracts for futures trading
- Web3 integration for contract interaction

## Improvements Made

1. Fixed backend issues:
   - Created proper requirements.txt with all necessary dependencies
   - Fixed import paths in the backend services
   - Created an ABI JSON file for blockchain integration
   - Implemented a volatility model for predictions
   - Enhanced error handling in the API endpoints

2. Created a modern UI frontend:
   - Implemented a responsive design with styled-components
   - Created a proper directory structure for scalability
   - Developed dashboard, trading, portfolio, and analytics pages
   - Added interactive charts and data visualization components
   - Implemented state management with React Context API

3. Integrated frontend with backend:
   - Created API service utilities for communication
   - Implemented proper error handling for API requests
   - Set up global state management for data sharing

4. Deployed the website:
   - The application is deployed at: https://hzuesgxl.manus.space

## Getting Started

### Backend Setup
1. Navigate to the backend directory: `cd code/backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn app:app --host 0.0.0.0 --port 8000`

### Frontend Setup
1. Navigate to the frontend directory: `cd code/frontend`
2. Install dependencies: `npm install`
3. Start development server: `npm start`
4. Build for production: `npm run build`
