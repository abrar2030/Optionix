# Enhanced Quantitative Models

This directory contains enhanced quantitative models for the Optionix platform, providing advanced volatility calculations and market calibration capabilities.

## Contents

### Stochastic Volatility Models

- `stochastic_volatility.py`: Implementation of advanced stochastic volatility models including:
    - Heston model with full calibration capabilities
    - SABR (Stochastic Alpha Beta Rho) model
    - Path simulation methods
    - Implied volatility surface generation

### Local Volatility Models

- `local_volatility.py`: Implementation of Dupire's local volatility model with:
    - Volatility surface construction
    - Numerical methods for partial differential equations
    - Calibration to market option prices

### Volatility Surface Construction

- `volatility_surface.py`: Tools for building and managing volatility surfaces with:
    - Interpolation methods (cubic spline, SVI parameterization)
    - Arbitrage-free constraints
    - Term structure modeling
    - Surface visualization tools

### Market Calibration Framework

- `calibration_engine.py`: Core engine for calibrating models to market data with:
    - Optimization algorithms
    - Objective function definitions
    - Parameter constraints
    - Multi-asset calibration

### Market Data Management

- `market_data.py`: Tools for managing and processing market data with:
    - Data cleaning and normalization
    - Implied volatility calculation
    - Forward curve construction
    - Historical data analysis

### Model Validation

- `model_validation.py`: Framework for validating calibrated models with:
    - Backtesting framework
    - Error metrics calculation
    - Sensitivity analysis
    - Stress testing for model parameters

## Usage

These enhanced models can be imported and used as follows:

```python
# Example for Heston model
from enhanced.stochastic_volatility import HestonModel

# Initialize model
heston = HestonModel()

# Simulate price paths
S0 = 100  # Initial price
r = 0.02  # Risk-free rate
T = 1.0   # Time horizon (1 year)
paths, variances = heston.simulate_paths(S0, r, T)

# Price an option
K = 100  # Strike price
option_price = heston.price_option(S0, K, T, r, option_type='call')

# Calibrate to market data
market_data = [
    {'S': 100, 'K': 95, 'T': 0.5, 'r': 0.02, 'price': 10.5, 'type': 'call'},
    {'S': 100, 'K': 100, 'T': 0.5, 'r': 0.02, 'price': 7.2, 'type': 'call'},
    {'S': 100, 'K': 105, 'T': 0.5, 'r': 0.02, 'price': 4.8, 'type': 'call'}
]
calibrated_params = heston.calibrate(market_data)
```

## Integration

These models integrate with the existing Optionix pricing engine and can be used for:

- Advanced option pricing
- Risk management
- Volatility trading strategies
- Market making
- Hedging

## Dependencies

- NumPy
- SciPy
- Matplotlib (for visualization functions)
