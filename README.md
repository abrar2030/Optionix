# Optionix Project Documentation

[![CI Status](https://img.shields.io/github/workflow/status/abrar2030/Optionix/CI/main?label=CI)](https://github.com/abrar2030/Optionix/actions)
[![Test Coverage](https://img.shields.io/codecov/c/github/abrar2030/Optionix/main?label=Coverage)](https://codecov.io/gh/abrar2030/Optionix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This document provides an overview of the Optionix project structure, components, and improvements made.

<div align="center">
  <img src="docs/Optionix.bmp" alt="Trading Derivatives Platform" width="100%">
</div>

> **Note**: This Project is currently under active development. Features and functionalities are being added and improved continuously to enhance user experience.

## Table of Contents
- [Project Structure](#project-structure)
- [Components](#components)
- [Feature Implementation Status](#feature-implementation-status)
- [Improvements Made](#improvements-made)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

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

## Feature Implementation Status

| Feature | Status | Description | Planned Release |
|---------|--------|-------------|----------------|
| **Backend Services** |
| Options Pricing API | ✅ Implemented | Core pricing engine for options | v1.0 |
| Volatility Prediction | ✅ Implemented | AI-based volatility forecasting | v1.0 |
| Blockchain Integration | ✅ Implemented | Smart contract interaction | v1.0 |
| User Authentication | ✅ Implemented | Secure user login and registration | v1.0 |
| Market Data API | 🔄 In Progress | Real-time market data integration | v1.1 |
| Order Execution | 🔄 In Progress | Trade execution and management | v1.1 |
| Risk Management API | 📅 Planned | Portfolio risk assessment | v1.2 |
| **Frontend Components** |
| Responsive Dashboard | ✅ Implemented | Main user interface | v1.0 |
| Market Overview | ✅ Implemented | Market summary and trends | v1.0 |
| Portfolio Summary | ✅ Implemented | User portfolio tracking | v1.0 |
| Option Chain | ✅ Implemented | Options listing and details | v1.0 |
| Order Book | ✅ Implemented | Current market orders | v1.0 |
| Position Tracking | 🔄 In Progress | Real-time position monitoring | v1.1 |
| Risk Assessment | 🔄 In Progress | Portfolio risk visualization | v1.1 |
| Volatility Charts | ✅ Implemented | Historical and predicted volatility | v1.0 |
| Strategy Builder | 📅 Planned | Custom options strategy creation | v1.2 |
| **AI Models** |
| Volatility Prediction | ✅ Implemented | ML model for volatility forecasting | v1.0 |
| Price Movement Prediction | 🔄 In Progress | Asset price direction forecasting | v1.1 |
| Sentiment Analysis | 📅 Planned | News and social media analysis | v1.2 |
| Options Strategy Recommendation | 📅 Planned | AI-based strategy suggestions | v1.2 |
| **Blockchain Integration** |
| Futures Contracts | ✅ Implemented | Smart contracts for futures trading | v1.0 |
| Web3 Integration | ✅ Implemented | Frontend-blockchain connection | v1.0 |
| Decentralized Settlement | 🔄 In Progress | P2P settlement mechanism | v1.1 |
| On-chain Options | 📅 Planned | Fully on-chain options contracts | v1.2 |

**Legend:**
- ✅ Implemented: Feature is complete and available
- 🔄 In Progress: Feature is currently being developed
- 📅 Planned: Feature is planned for future release

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

## Testing

The project includes comprehensive testing to ensure reliability and accuracy:

### Backend Testing
- Unit tests for API endpoints using pytest
- Integration tests for blockchain interaction
- Performance tests for options pricing algorithms

### Frontend Testing
- Component tests using React Testing Library
- End-to-end tests with Cypress
- State management tests

### AI Model Testing
- Model accuracy validation
- Backtesting against historical data
- Performance benchmarking

To run tests:

```bash
# Backend tests
cd code/backend
pytest

# Frontend tests
cd code/frontend
npm test

# AI model tests
cd code/ai_models
python -m unittest discover

# Run all tests with the convenience script
./test_backend.sh
```

## CI/CD Pipeline

Optionix uses GitHub Actions for continuous integration and deployment:

### Continuous Integration
- Automated testing on each pull request and push to main
- Code quality checks with ESLint, Prettier, and Pylint
- Test coverage reporting with pytest-cov and Jest
- Security scanning for vulnerabilities

### Continuous Deployment
- Automated deployment to staging environment on merge to main
- Manual promotion to production after approval
- Docker image building and publishing
- Infrastructure updates via Terraform

Current CI/CD Status:
- Build: ![Build Status](https://img.shields.io/github/workflow/status/abrar2030/Optionix/CI/main?label=build)
- Test Coverage: ![Coverage](https://img.shields.io/codecov/c/github/abrar2030/Optionix/main?label=coverage)
- Code Quality: ![Code Quality](https://img.shields.io/codacy/grade/abrar2030/Optionix?label=code%20quality)

## Contributing

We welcome contributions to improve Optionix! Here's how you can contribute:

1. **Fork the repository**
   - Create your own copy of the project to work on

2. **Create a feature branch**
   - `git checkout -b feature/amazing-feature`
   - Use descriptive branch names that reflect the changes

3. **Make your changes**
   - Follow the coding standards and guidelines
   - Write clean, maintainable, and tested code
   - Update documentation as needed

4. **Commit your changes**
   - `git commit -m 'Add some amazing feature'`
   - Use clear and descriptive commit messages
   - Reference issue numbers when applicable

5. **Push to branch**
   - `git push origin feature/amazing-feature`

6. **Open Pull Request**
   - Provide a clear description of the changes
   - Link to any relevant issues
   - Respond to review comments and make necessary adjustments

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Use ESLint and Prettier for JavaScript/React code
- Write unit tests for new features
- Update documentation for any changes
- Ensure all tests pass before submitting a pull request
- Keep pull requests focused on a single feature or fix

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
