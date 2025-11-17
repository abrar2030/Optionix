"""
Market Calibration Framework for option pricing models.

This module implements calibration engines for various option pricing models
to ensure they match market prices.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CalibrationEngine:
    """
    Implementation of calibration engine for option pricing models.

    This class provides methods to calibrate various option pricing models
    to market data.
    """

    def __init__(self):
        """
        Initialize calibration engine.
        """
        pass

    def calibrate_heston(self, option_data, initial_params=None):
        """
        Calibrate Heston model parameters to market data.

        Args:
            option_data (dict or list): Option data
                If dict: Expected format is {"calls": [...], "puts": [...]}
                If list: List of option data dictionaries
            initial_params (dict, optional): Initial parameters for calibration

        Returns:
            dict: Calibrated parameters
        """
        # Import here to avoid circular imports
        from quantitative.enhanced.stochastic_volatility import HestonModel

        # Create Heston model
        heston = HestonModel(initial_params)

        # Extract option data
        options = []

        if isinstance(option_data, dict):
            # Extract from dict format
            if "calls" in option_data:
                for call in option_data["calls"]:
                    options.append(
                        {
                            "strike": call["strike"],
                            "expiry": call.get(
                                "expiry", 0.5
                            ),  # Default to 6 months if not specified
                            "price": (
                                call["price"]
                                if "price" in call
                                else self._bs_price_from_iv(
                                    call["iv"], call["strike"], 0.5, "call"
                                )
                            ),
                            "option_type": "call",
                        }
                    )
            if "puts" in option_data:
                for put in option_data["puts"]:
                    options.append(
                        {
                            "strike": put["strike"],
                            "expiry": put.get(
                                "expiry", 0.5
                            ),  # Default to 6 months if not specified
                            "price": (
                                put["price"]
                                if "price" in put
                                else self._bs_price_from_iv(
                                    put["iv"], put["strike"], 0.5, "put"
                                )
                            ),
                            "option_type": "put",
                        }
                    )
        else:
            # Assume list format
            options = option_data

        # Calibrate model
        calibrated_params = heston.calibrate(options)

        return calibrated_params

    def calibrate_sabr(self, option_data, initial_params=None):
        """
        Calibrate SABR model parameters to market data.

        Args:
            option_data (dict or list): Option data
                If dict: Expected format is {"calls": [...], "puts": [...]}
                If list: List of option data dictionaries
            initial_params (dict, optional): Initial parameters for calibration

        Returns:
            dict: Calibrated parameters
        """
        # Import here to avoid circular imports
        from quantitative.enhanced.stochastic_volatility import SabrModel

        # Create SABR model
        sabr = SabrModel(initial_params)

        # Extract option data
        options = []

        if isinstance(option_data, dict):
            # Extract from dict format
            spot = 450.0  # Default spot price

            if "calls" in option_data:
                for call in option_data["calls"]:
                    options.append(
                        {
                            "strike": call["strike"],
                            "forward": call.get("forward", spot),
                            "time_to_expiry": call.get(
                                "expiry", 0.5
                            ),  # Default to 6 months if not specified
                            "market_vol": call["iv"],
                        }
                    )
            if "puts" in option_data:
                for put in option_data["puts"]:
                    options.append(
                        {
                            "strike": put["strike"],
                            "forward": put.get("forward", spot),
                            "time_to_expiry": put.get(
                                "expiry", 0.5
                            ),  # Default to 6 months if not specified
                            "market_vol": put["iv"],
                        }
                    )
        else:
            # Assume list format
            options = option_data

        # Calibrate model
        calibrated_params = sabr.calibrate(options)

        return calibrated_params

    def calibrate_local_volatility(self, option_data, spot, rate, dividend=0):
        """
        Calibrate local volatility model to market data.

        Args:
            option_data (dict or list): Option data
                If dict: Expected format is {"calls": [...], "puts": [...]}
                If list: List of option data dictionaries
            spot (float): Spot price
            rate (float): Risk-free interest rate
            dividend (float): Dividend yield

        Returns:
            tuple: (local_vol_surface, strike_grid, time_grid)
        """
        # Import here to avoid circular imports
        from quantitative.enhanced.local_volatility import DupireLocalVolModel

        # Create local volatility model
        dupire = DupireLocalVolModel()

        # Extract option data
        options = []

        if isinstance(option_data, dict):
            # Extract from dict format
            if "calls" in option_data:
                for call in option_data["calls"]:
                    options.append(
                        {
                            "strike": call["strike"],
                            "expiry": call.get(
                                "expiry", 0.5
                            ),  # Default to 6 months if not specified
                            "price": (
                                call["price"]
                                if "price" in call
                                else self._bs_price_from_iv(
                                    call["iv"],
                                    call["strike"],
                                    0.5,
                                    "call",
                                    spot,
                                    rate,
                                    dividend,
                                )
                            ),
                            "option_type": "call",
                        }
                    )
            if "puts" in option_data:
                for put in option_data["puts"]:
                    options.append(
                        {
                            "strike": put["strike"],
                            "expiry": put.get(
                                "expiry", 0.5
                            ),  # Default to 6 months if not specified
                            "price": (
                                put["price"]
                                if "price" in put
                                else self._bs_price_from_iv(
                                    put["iv"],
                                    put["strike"],
                                    0.5,
                                    "put",
                                    spot,
                                    rate,
                                    dividend,
                                )
                            ),
                            "option_type": "put",
                        }
                    )
        else:
            # Assume list format
            options = option_data

        # Calibrate model
        local_vol_surface, strike_grid, time_grid = dupire.calibrate(
            options, spot, rate, dividend
        )

        return local_vol_surface, strike_grid, time_grid

    def _bs_price_from_iv(
        self,
        iv,
        strike,
        time_to_expiry,
        option_type,
        spot=450.0,
        rate=0.02,
        dividend=0.01,
    ):
        """
        Calculate Black-Scholes price from implied volatility.

        Args:
            iv (float): Implied volatility
            strike (float): Strike price
            time_to_expiry (float): Time to expiry in years
            option_type (str): Option type ('call' or 'put')
            spot (float): Spot price
            rate (float): Risk-free interest rate
            dividend (float): Dividend yield

        Returns:
            float: Option price
        """
        from scipy import stats

        # Calculate d1 and d2
        d1 = (
            np.log(spot / strike) + (rate - dividend + 0.5 * iv**2) * time_to_expiry
        ) / (iv * np.sqrt(time_to_expiry))
        d2 = d1 - iv * np.sqrt(time_to_expiry)

        # Calculate price
        if option_type.lower() == "call":
            price = spot * np.exp(-dividend * time_to_expiry) * stats.norm.cdf(
                d1
            ) - strike * np.exp(-rate * time_to_expiry) * stats.norm.cdf(d2)
        else:  # put
            price = strike * np.exp(-rate * time_to_expiry) * stats.norm.cdf(
                -d2
            ) - spot * np.exp(-dividend * time_to_expiry) * stats.norm.cdf(-d1)

        return price
