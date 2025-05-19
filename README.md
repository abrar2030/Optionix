# Optionix

[![CI/CD Status](https://img.shields.io/github/actions/workflow/status/abrar2030/Optionix/ci-cd.yml?branch=main&label=CI/CD&logo=github)](https://github.com/abrar2030/Optionix/actions)
[![Test Coverage](https://img.shields.io/codecov/c/github/abrar2030/Optionix/main?label=Coverage)](https://codecov.io/gh/abrar2030/Optionix)
[![License](https://img.shields.io/github/license/abrar2030/Optionix)](https://github.com/abrar2030/Optionix/blob/main/LICENSE)

## 📈 Options Trading & Analytics Platform

Optionix is a comprehensive options trading and analytics platform that combines traditional finance with blockchain technology. The platform provides advanced options pricing models, real-time market data, and AI-powered trading signals to help traders make informed decisions.

<div align="center">
  <img src="docs/images/Optionix_dashboard.bmp" alt="Optionix Trading Dashboard" width="80%">
</div>

> **Note**: This project is under active development. Features and functionalities are continuously being enhanced to improve user experience and trading capabilities.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Feature Implementation Status](#feature-implementation-status)
- [Improvements Made](#improvements-made)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

## Overview

Optionix is a next-generation options trading platform that leverages advanced algorithms, machine learning, and blockchain technology to provide traders with powerful tools for options analysis and trading. The platform includes:

- **Options Pricing Engine**: Advanced mathematical models for accurate options pricing
- **Market Data Integration**: Real-time market data for informed decision-making
- **AI Trading Signals**: Machine learning algorithms to identify trading opportunities
- **Portfolio Management**: Comprehensive tools for tracking and managing options positions
- **Blockchain Integration**: Smart contracts for decentralized options trading
- **Risk Analysis**: Sophisticated risk assessment and visualization tools

## Features

### Options Trading
- Real-time options chain data
- Multi-leg strategy builder
- One-click trade execution
- Position tracking and P&L analysis
- Historical performance metrics

### Analytics
- Volatility surface visualization
- Greeks calculation and visualization
- Implied volatility analysis
- Options strategy payoff diagrams
- Risk/reward ratio calculations

### AI-Powered Insights
- Volatility prediction models
- Options mispricing detection
- Market sentiment analysis
- Automated trading signals
- Personalized strategy recommendations

### Blockchain Integration
- Decentralized options contracts
- Smart contract settlement
- On-chain position verification
- Cross-chain asset collateralization
- Transparent transaction history

## Technology Stack

### Backend
- **Language**: Python, Rust (for performance-critical components)
- **Framework**: FastAPI
- **Database**: PostgreSQL, TimescaleDB (for time-series data)
- **Caching**: Redis
- **Message Queue**: RabbitMQ
- **ML Framework**: PyTorch, scikit-learn
- **Blockchain**: Ethereum, Solidity

### Frontend
- **Framework**: React with TypeScript
- **State Management**: Redux Toolkit
- **Styling**: Styled Components, TailwindCSS
- **Data Visualization**: D3.js, TradingView Charts
- **Web3**: ethers.js

### Mobile App
- **Framework**: React Native
- **State Management**: Redux Toolkit
- **Navigation**: React Navigation
- **UI Components**: React Native Paper
- **Charts**: Victory Native

### DevOps
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Logging**: ELK Stack

## Feature Implementation Status

| Feature | Status | Description | Planned Release |
|---------|--------|-------------|----------------|
| **Options Trading** |
| Options Chain Data | ✅ Implemented | Real-time options data display | v1.0 |
| Strategy Builder | ✅ Implemented | Multi-leg strategy creation | v1.0 |
| Trade Execution | ✅ Implemented | Order placement and management | v1.0 |
| Position Tracking | ✅ Implemented | Portfolio position monitoring | v1.0 |
| Historical Analysis | 🔄 In Progress | Past performance metrics | v1.1 |
| **Analytics** |
| Volatility Surface | ✅ Implemented | 3D visualization of implied volatility | v1.0 |
| Greeks Calculation | ✅ Implemented | Delta, gamma, theta, vega, rho | v1.0 |
| Payoff Diagrams | ✅ Implemented | Strategy profit/loss visualization | v1.0 |
| Risk Analysis | 🔄 In Progress | Advanced risk metrics | v1.1 |
| Scenario Testing | 📅 Planned | What-if analysis for strategies | v1.2 |
| **AI Features** |
| Volatility Prediction | ✅ Implemented | ML-based volatility forecasting | v1.0 |
| Mispricing Detection | ✅ Implemented | Identify undervalued options | v1.0 |
| Trading Signals | 🔄 In Progress | Automated buy/sell recommendations | v1.1 |
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

### Using the Setup Script

For a quick setup of the entire application:

```bash
# Clone the repository
git clone https://github.com/abrar2030/Optionix.git
cd Optionix

# Run the setup script
./setup_optionix_env.sh

# Start the application
./run_optionix.sh
```

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
- Build: ![Build Status](https://img.shields.io/github/actions/workflow/status/abrar2030/Optionix/ci-cd.yml?branch=main&label=build)
- Test Coverage: ![Coverage](https://img.shields.io/codecov/c/github/abrar2030/Optionix/main?label=coverage)
- Code Quality: ![Code Quality](https://img.shields.io/lgtm/grade/javascript/g/abrar2030/Optionix?label=code%20quality)

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
