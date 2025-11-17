"""
Stochastic Volatility Models for option pricing.

This module implements advanced stochastic volatility models including
Heston model and SABR model for more accurate option pricing.
"""

import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HestonModel:
    """
    Implementation of the Heston stochastic volatility model.

    The Heston model assumes that the volatility of the asset follows a CIR process,
    allowing for mean-reversion in volatility and correlation between asset returns
    and volatility changes.
    """

    def __init__(self, params=None):
        """
        Initialize Heston model with parameters.

        Args:
            params (dict, optional): Model parameters
                - v0: Initial variance
                - kappa: Rate of mean reversion
                - theta: Long-run variance
                - sigma: Volatility of volatility
                - rho: Correlation between asset returns and variance
        """
        # Default parameters
        self.params = params or {
            "v0": 0.04,  # Initial variance
            "kappa": 2.0,  # Rate of mean reversion
            "theta": 0.04,  # Long-run variance
            "sigma": 0.3,  # Volatility of volatility
            "rho": -0.7,  # Correlation between asset returns and variance
        }

    def price_option(
        self,
        spot,
        strike,
        time_to_expiry,
        rate,
        dividend,
        option_type="call",
        method="monte_carlo",
        num_paths=10000,
        num_steps=100,
    ):
        """
        Price an option using the Heston model.

        Args:
            spot (float): Spot price of the underlying
            strike (float): Strike price of the option
            time_to_expiry (float): Time to expiry in years
            rate (float): Risk-free interest rate
            dividend (float): Dividend yield
            option_type (str): Option type ('call' or 'put')
            method (str): Pricing method ('monte_carlo' or 'semi_analytical')
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps

        Returns:
            float: Option price
        """
        if method == "monte_carlo":
            return self._price_monte_carlo(
                spot,
                strike,
                time_to_expiry,
                rate,
                dividend,
                option_type,
                num_paths,
                num_steps,
            )
        elif method == "semi_analytical":
            return self._price_semi_analytical(
                spot, strike, time_to_expiry, rate, dividend, option_type
            )
        else:
            raise ValueError(f"Unknown pricing method: {method}")

    def _price_monte_carlo(
        self,
        spot,
        strike,
        time_to_expiry,
        rate,
        dividend,
        option_type,
        num_paths,
        num_steps,
    ):
        """
        Price an option using Monte Carlo simulation.

        Args:
            spot (float): Spot price of the underlying
            strike (float): Strike price of the option
            time_to_expiry (float): Time to expiry in years
            rate (float): Risk-free interest rate
            dividend (float): Dividend yield
            option_type (str): Option type ('call' or 'put')
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps

        Returns:
            float: Option price
        """
        # Extract parameters
        v0 = self.params["v0"]
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        sigma = self.params["sigma"]
        rho = self.params["rho"]

        # Time step
        dt = time_to_expiry / num_steps

        # Initialize arrays
        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))

        # Set initial values
        S[:, 0] = spot
        v[:, 0] = v0

        # Generate correlated random numbers
        z1 = np.random.normal(0, 1, (num_paths, num_steps))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(
            0, 1, (num_paths, num_steps)
        )

        # Simulate paths
        for i in range(num_steps):
            # Ensure variance is non-negative
            v[:, i] = np.maximum(v[:, i], 0)

            # Update stock price
            S[:, i + 1] = S[:, i] * np.exp(
                (rate - dividend - 0.5 * v[:, i]) * dt
                + np.sqrt(v[:, i] * dt) * z1[:, i]
            )

            # Update variance using full truncation scheme
            v_next = (
                v[:, i]
                + kappa * (theta - v[:, i]) * dt
                + sigma * np.sqrt(v[:, i] * dt) * z2[:, i]
            )
            v[:, i + 1] = np.maximum(v_next, 0)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(S[:, -1] - strike, 0)
        else:  # put
            payoffs = np.maximum(strike - S[:, -1], 0)

        # Discount payoffs
        option_price = np.exp(-rate * time_to_expiry) * np.mean(payoffs)

        return option_price

    def _price_semi_analytical(
        self, spot, strike, time_to_expiry, rate, dividend, option_type
    ):
        """
        Price an option using semi-analytical method (Fourier transform).

        Args:
            spot (float): Spot price of the underlying
            strike (float): Strike price of the option
            time_to_expiry (float): Time to expiry in years
            rate (float): Risk-free interest rate
            dividend (float): Dividend yield
            option_type (str): Option type ('call' or 'put')

        Returns:
            float: Option price
        """
        # In a real implementation, this would use characteristic function and Fourier transform
        # For simplicity, we'll use a placeholder implementation

        # Extract parameters
        v0 = self.params["v0"]
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        sigma = self.params["sigma"]
        rho = self.params["rho"]

        # Calculate Black-Scholes price as a starting point
        forward = spot * np.exp((rate - dividend) * time_to_expiry)
        vol = np.sqrt(v0)
        d1 = (np.log(forward / strike) + 0.5 * vol**2 * time_to_expiry) / (
            vol * np.sqrt(time_to_expiry)
        )
        d2 = d1 - vol * np.sqrt(time_to_expiry)

        if option_type.lower() == "call":
            bs_price = np.exp(-rate * time_to_expiry) * (
                forward * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)
            )
        else:  # put
            bs_price = np.exp(-rate * time_to_expiry) * (
                strike * stats.norm.cdf(-d2) - forward * stats.norm.cdf(-d1)
            )

        # Apply a correction factor based on Heston parameters
        # This is a simplified approximation
        vol_of_vol_effect = 1 + 0.1 * sigma * np.sqrt(time_to_expiry)
        mean_reversion_effect = 1 - 0.05 * kappa * time_to_expiry
        correlation_effect = 1 + 0.1 * rho * np.sqrt(time_to_expiry)

        correction = vol_of_vol_effect * mean_reversion_effect * correlation_effect

        return bs_price * correction

    def calibrate(self, option_data, initial_params=None, bounds=None, method="SLSQP"):
        """
        Calibrate model parameters to market data.

        Args:
            option_data (list): List of option data dictionaries
                Each dictionary should contain:
                - spot: Spot price
                - strike: Strike price
                - time_to_expiry: Time to expiry in years
                - rate: Risk-free rate
                - dividend: Dividend yield
                - option_type: Option type ('call' or 'put')
                - market_price: Market price of the option
            initial_params (dict, optional): Initial parameters for calibration
            bounds (dict, optional): Parameter bounds for calibration
            method (str): Optimization method

        Returns:
            dict: Calibrated parameters
        """
        # Set initial parameters
        if initial_params is None:
            initial_params = self.params.copy()

        # Set parameter bounds
        if bounds is None:
            bounds = {
                "v0": (0.001, 0.5),
                "kappa": (0.1, 10.0),
                "theta": (0.001, 0.5),
                "sigma": (0.01, 2.0),
                "rho": (-0.99, 0.99),
            }

        # Convert parameters to array for optimization
        param_keys = ["v0", "kappa", "theta", "sigma", "rho"]
        initial_values = [initial_params[key] for key in param_keys]
        param_bounds = [bounds[key] for key in param_keys]

        # Define objective function
        def objective(params):
            # Update model parameters
            param_dict = {key: params[i] for i, key in enumerate(param_keys)}
            self.params = param_dict

            # Calculate sum of squared errors
            sse = 0
            for option in option_data:
                model_price = self.price_option(
                    option["spot"],
                    option["strike"],
                    option["time_to_expiry"],
                    option["rate"],
                    option["dividend"],
                    option["option_type"],
                    method="semi_analytical",  # Use faster method for calibration
                )
                sse += (model_price - option["market_price"]) ** 2

            return sse

        # Run optimization
        result = minimize(
            objective,
            initial_values,
            method=method,
            bounds=param_bounds,
            options={"maxiter": 100},
        )

        # Update model parameters
        self.params = {key: result.x[i] for i, key in enumerate(param_keys)}

        return self.params

    def simulate_paths(
        self, spot, time_horizon, rate, dividend, num_paths=1, num_steps=252
    ):
        """
        Simulate asset price and variance paths.

        Args:
            spot (float): Initial spot price
            time_horizon (float): Time horizon in years
            rate (float): Risk-free interest rate
            dividend (float): Dividend yield
            num_paths (int): Number of paths to simulate
            num_steps (int): Number of time steps

        Returns:
            tuple: (price_paths, variance_paths, time_grid)
        """
        # Extract parameters
        v0 = self.params["v0"]
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        sigma = self.params["sigma"]
        rho = self.params["rho"]

        # Time step
        dt = time_horizon / num_steps

        # Initialize arrays
        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))
        time_grid = np.linspace(0, time_horizon, num_steps + 1)

        # Set initial values
        S[:, 0] = spot
        v[:, 0] = v0

        # Generate correlated random numbers
        z1 = np.random.normal(0, 1, (num_paths, num_steps))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(
            0, 1, (num_paths, num_steps)
        )

        # Simulate paths
        for i in range(num_steps):
            # Ensure variance is non-negative
            v[:, i] = np.maximum(v[:, i], 0)

            # Update stock price
            S[:, i + 1] = S[:, i] * np.exp(
                (rate - dividend - 0.5 * v[:, i]) * dt
                + np.sqrt(v[:, i] * dt) * z1[:, i]
            )

            # Update variance using full truncation scheme
            v_next = (
                v[:, i]
                + kappa * (theta - v[:, i]) * dt
                + sigma * np.sqrt(v[:, i] * dt) * z2[:, i]
            )
            v[:, i + 1] = np.maximum(v_next, 0)

        return S, v, time_grid


class SabrModel:
    """
    Implementation of the SABR (Stochastic Alpha Beta Rho) model.

    The SABR model is a stochastic volatility model that captures the volatility smile
    and is particularly popular for interest rate derivatives.
    """

    def __init__(self, params=None):
        """
        Initialize SABR model with parameters.

        Args:
            params (dict, optional): Model parameters
                - alpha: Volatility of volatility
                - beta: CEV parameter (0 <= beta <= 1)
                - rho: Correlation between asset and volatility
                - nu: Volatility of volatility parameter
        """
        # Default parameters
        self.params = params or {
            "alpha": 0.3,  # Initial volatility
            "beta": 0.7,  # CEV parameter (0 <= beta <= 1)
            "rho": -0.5,  # Correlation between asset and volatility
            "nu": 0.4,  # Volatility of volatility
        }

    def implied_volatility(self, strike, forward, time_to_expiry, params=None):
        """
        Calculate SABR implied volatility.

        Args:
            strike (float): Strike price
            forward (float): Forward price
            time_to_expiry (float): Time to expiry in years
            params (dict, optional): Model parameters (uses instance parameters if None)

        Returns:
            float: SABR implied volatility
        """
        # Use provided parameters or instance parameters
        if params is None:
            params = self.params

        # Extract parameters
        alpha = params["alpha"]
        beta = params["beta"]
        rho = params["rho"]
        nu = params["nu"]

        # Handle ATM case separately
        if abs(strike - forward) < 1e-10:
            return self._atm_implied_volatility(
                forward, time_to_expiry, alpha, beta, rho, nu
            )

        # Calculate SABR implied volatility using Hagan's formula
        F = forward
        K = strike
        T = time_to_expiry

        # Calculate intermediate terms
        z = nu / alpha * (F * K) ** (0.5 * (1 - beta)) * np.log(F / K)
        x = np.log(F / K)

        # Handle small z case
        if abs(z) < 1e-6:
            z_term = 1
        else:
            z_term = z / np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        # Calculate volatility
        numerator = alpha * ((F * K) ** (0.5 * (1 - beta)))
        denominator = (
            (F ** (1 - beta) - K ** (1 - beta)) / (1 - beta)
            if beta != 1
            else np.log(F / K)
        )

        vol = numerator * z_term / denominator

        # Apply correction terms
        correction1 = 1 + ((1 - beta) ** 2 / 24) * (alpha**2 / ((F * K) ** (1 - beta)))
        correction2 = 1 + (1 / 4) * rho * beta * nu * alpha / (
            (F * K) ** ((1 - beta) / 2)
        )
        correction3 = 1 + ((2 - 3 * rho**2) / 24) * nu**2

        vol *= correction1 * correction2 * correction3

        return vol

    def _atm_implied_volatility(self, forward, time_to_expiry, alpha, beta, rho, nu):
        """
        Calculate ATM SABR implied volatility.

        Args:
            forward (float): Forward price
            time_to_expiry (float): Time to expiry in years
            alpha (float): Alpha parameter
            beta (float): Beta parameter
            rho (float): Rho parameter
            nu (float): Nu parameter

        Returns:
            float: ATM SABR implied volatility
        """
        F = forward
        T = time_to_expiry

        # ATM volatility formula
        vol = alpha / (F ** (1 - beta))

        # Apply correction terms
        correction1 = 1 + ((1 - beta) ** 2 / 24) * (alpha**2 / (F ** (2 - 2 * beta)))
        correction2 = 1 + (1 / 4) * rho * beta * nu * alpha / (F ** (1 - beta))
        correction3 = 1 + ((2 - 3 * rho**2) / 24) * nu**2

        vol *= correction1 * correction2 * correction3

        return vol

    def calibrate(self, option_data, initial_params=None, bounds=None, method="SLSQP"):
        """
        Calibrate SABR parameters to market data.

        Args:
            option_data (list): List of option data dictionaries
                Each dictionary should contain:
                - forward: Forward price
                - strike: Strike price
                - time_to_expiry: Time to expiry in years
                - market_vol: Market implied volatility
            initial_params (dict, optional): Initial parameters for calibration
            bounds (dict, optional): Parameter bounds for calibration
            method (str): Optimization method

        Returns:
            dict: Calibrated parameters
        """
        # Set initial parameters
        if initial_params is None:
            initial_params = self.params.copy()

        # Set parameter bounds
        if bounds is None:
            bounds = {
                "alpha": (0.01, 1.0),
                "beta": (0.01, 0.99),
                "rho": (-0.99, 0.99),
                "nu": (0.01, 1.0),
            }

        # Convert parameters to array for optimization
        param_keys = ["alpha", "beta", "rho", "nu"]
        initial_values = [initial_params[key] for key in param_keys]
        param_bounds = [bounds[key] for key in param_keys]

        # Define objective function
        def objective(params):
            # Update model parameters
            param_dict = {key: params[i] for i, key in enumerate(param_keys)}

            # Calculate sum of squared errors
            sse = 0
            for option in option_data:
                model_vol = self.implied_volatility(
                    option["strike"],
                    option["forward"],
                    option["time_to_expiry"],
                    param_dict,
                )
                sse += (model_vol - option["market_vol"]) ** 2

            return sse

        # Run optimization
        result = minimize(
            objective,
            initial_values,
            method=method,
            bounds=param_bounds,
            options={"maxiter": 100},
        )

        # Update model parameters
        self.params = {key: result.x[i] for i, key in enumerate(param_keys)}

        return self.params

    def price_option(self, strike, forward, time_to_expiry, rate, option_type="call"):
        """
        Price an option using the SABR model.

        Args:
            strike (float): Strike price
            forward (float): Forward price
            time_to_expiry (float): Time to expiry in years
            rate (float): Risk-free interest rate
            option_type (str): Option type ('call' or 'put')

        Returns:
            float: Option price
        """
        # Calculate implied volatility
        implied_vol = self.implied_volatility(strike, forward, time_to_expiry)

        # Calculate option price using Black formula
        discount = np.exp(-rate * time_to_expiry)

        d1 = (np.log(forward / strike) + 0.5 * implied_vol**2 * time_to_expiry) / (
            implied_vol * np.sqrt(time_to_expiry)
        )
        d2 = d1 - implied_vol * np.sqrt(time_to_expiry)

        if option_type.lower() == "call":
            price = discount * (
                forward * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)
            )
        else:  # put
            price = discount * (
                strike * stats.norm.cdf(-d2) - forward * stats.norm.cdf(-d1)
            )

        return price

    def plot_volatility_smile(self, forward, strikes, time_to_expiry, title=None):
        """
        Plot the volatility smile for a given forward and time to expiry.

        Args:
            forward (float): Forward price
            strikes (array): Array of strike prices
            time_to_expiry (float): Time to expiry in years
            title (str, optional): Plot title

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Calculate implied volatilities
        implied_vols = [
            self.implied_volatility(K, forward, time_to_expiry) for K in strikes
        ]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(strikes, implied_vols, "b-", linewidth=2)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"SABR Volatility Smile (F={forward}, T={time_to_expiry})")

        ax.grid(True)

        return fig
