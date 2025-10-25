# Optionix

[![CI/CD Status](https://img.shields.io/github/actions/workflow/status/abrar2030/Optionix/ci-cd.yml?branch=main&label=CI/CD&logo=github)](https://github.com/abrar2030/Optionix/actions)
[![Test Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://github.com/abrar2030/Optionix/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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

* **Options Pricing Engine**: Advanced mathematical models for accurate options pricing
* **Market Data Integration**: Real-time market data for informed decision-making
* **AI Trading Signals**: Machine learning algorithms to identify trading opportunities
* **Portfolio Management**: Comprehensive tools for tracking and managing options positions
* **Blockchain Integration**: Smart contracts for decentralized options trading
* **Risk Analysis**: Sophisticated risk assessment and visualization tools

## Features

### Options Trading
* Real-time options chain data
* Multi-leg strategy builder
* One-click trade execution
* Position tracking and P&L analysis
* Historical performance metrics

### Analytics
* Volatility surface visualization
* Greeks calculation and visualization
* Implied volatility analysis
* Options strategy payoff diagrams
* Risk/reward ratio calculations

### AI Features
* Volatility prediction models
* Options mispricing detection
* Market sentiment analysis
* Automated trading signals
* Personalized strategy recommendations

### Blockchain Integration
* Decentralized options contracts
* Smart contract settlement
* On-chain position verification
* Cross-chain asset collateralization
* Transparent transaction history

## Technology Stack

### Backend
* **Language**: Python, Rust (for performance-critical components)
* **Framework**: FastAPI
* **Database**: PostgreSQL, TimescaleDB (for time-series data)
* **Caching**: Redis
* **Message Queue**: RabbitMQ
* **ML Framework**: PyTorch, scikit-learn
* **Blockchain**: Ethereum, Solidity

### Web Frontend
* **Framework**: React with TypeScript
* **State Management**: Redux Toolkit
* **Styling**: Styled Components, TailwindCSS
* **Data Visualization**: D3.js, TradingView Charts
* **Web3**: ethers.js

### Mobile Frontend
* **Framework**: React Native
* **State Management**: Redux Toolkit
* **Navigation**: React Navigation
* **UI Components**: React Native Paper
* **Charts**: Victory Native

### Infrastructure
* **Containerization**: Docker
* **Orchestration**: Kubernetes
* **CI/CD**: GitHub Actions
* **Monitoring**: Prometheus, Grafana
* **Logging**: ELK Stack

## Improvements Made

1. Fixed backend issues:
   * Created proper requirements.txt with all necessary dependencies
   * Fixed import paths in the backend services
   * Created an ABI JSON file for blockchain integration
   * Implemented a volatility model for predictions
   * Enhanced error handling in the API endpoints

2. Created a modern UI frontend:
   * Implemented a responsive design with styled-components
   * Created a proper directory structure for scalability
   * Developed dashboard, trading, portfolio, and analytics pages
   * Added interactive charts and data visualization components
   * Implemented state management with React Context API

3. Integrated frontend with backend:
   * Created API service utilities for communication
   * Implemented proper error handling for API requests
   * Set up global state management for data sharing

4. Deployed the website:
   * The application is deployed at: [https://hzuesgxl.manus.space](https://hzuesgxl.manus.space)

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

The project maintains comprehensive test coverage across all components to ensure reliability and accuracy.

### Test Coverage

| Component | Coverage | Status |
|-----------|----------|--------|
| Backend API | 85% | ✅ |
| Options Pricing Engine | 90% | ✅ |
| Frontend Components | 78% | ✅ |
| Blockchain Integration | 75% | ✅ |
| AI Models | 77% | ✅ |
| Overall | 81% | ✅ |

### Backend Testing
* Unit tests for API endpoints using pytest
* Integration tests for blockchain interaction
* Performance tests for options pricing algorithms

### Frontend Testing
* Component tests using React Testing Library
* End-to-end tests with Cypress
* State management tests

### AI Model Testing
* Model accuracy validation
* Backtesting against historical data
* Performance benchmarking

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

The project uses GitHub Actions for continuous integration and deployment:

* Automated testing on pull requests
* Code quality checks with ESLint and Pylint
* Docker image building and publishing
* Automated deployment to staging and production environments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.