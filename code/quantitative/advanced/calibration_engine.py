"""
Market Calibration Framework for option pricing models.

This module implements calibration engines for various option pricing models
to ensure they match market prices.
"""

import logging
import numpy as np

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

    def __init__(self) -> None:
        """
        Initialize calibration engine.
        """

    def calibrate_heston(self, option_data: Any, initial_params: Any = None) -> Any:
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
        from quantitative.advanced.stochastic_volatility import HestonModel

        heston = HestonModel(initial_params)
        options = []
        if isinstance(option_data, dict):
            if "calls" in option_data:
                for call in option_data["calls"]:
                    options.append(
                        {
                            "strike": call["strike"],
                            "expiry": call.get("expiry", 0.5),
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
                            "expiry": put.get("expiry", 0.5),
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
            options = option_data
        calibrated_params = heston.calibrate(options)
        return calibrated_params

    def calibrate_sabr(self, option_data: Any, initial_params: Any = None) -> Any:
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
        from quantitative.advanced.stochastic_volatility import SabrModel

        sabr = SabrModel(initial_params)
        options = []
        if isinstance(option_data, dict):
            spot = 450.0
            if "calls" in option_data:
                for call in option_data["calls"]:
                    options.append(
                        {
                            "strike": call["strike"],
                            "forward": call.get("forward", spot),
                            "time_to_expiry": call.get("expiry", 0.5),
                            "market_vol": call["iv"],
                        }
                    )
            if "puts" in option_data:
                for put in option_data["puts"]:
                    options.append(
                        {
                            "strike": put["strike"],
                            "forward": put.get("forward", spot),
                            "time_to_expiry": put.get("expiry", 0.5),
                            "market_vol": put["iv"],
                        }
                    )
        else:
            options = option_data
        calibrated_params = sabr.calibrate(options)
        return calibrated_params

    def calibrate_local_volatility(
        self, option_data: Any, spot: Any, rate: Any, dividend: Any = 0
    ) -> Any:
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
        from quantitative.advanced.local_volatility import DupireLocalVolModel

        dupire = DupireLocalVolModel()
        options = []
        if isinstance(option_data, dict):
            if "calls" in option_data:
                for call in option_data["calls"]:
                    options.append(
                        {
                            "strike": call["strike"],
                            "expiry": call.get("expiry", 0.5),
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
                            "expiry": put.get("expiry", 0.5),
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
            options = option_data
        local_vol_surface, strike_grid, time_grid = dupire.calibrate(
            options, spot, rate, dividend
        )
        return (local_vol_surface, strike_grid, time_grid)

    def _bs_price_from_iv(
        self,
        iv: Any,
        strike: Any,
        time_to_expiry: Any,
        option_type: Any,
        spot: Any = 450.0,
        rate: Any = 0.02,
        dividend: Any = 0.01,
    ) -> Any:
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

        d1 = (
            np.log(spot / strike) + (rate - dividend + 0.5 * iv**2) * time_to_expiry
        ) / (iv * np.sqrt(time_to_expiry))
        d2 = d1 - iv * np.sqrt(time_to_expiry)
        if option_type.lower() == "call":
            price = spot * np.exp(-dividend * time_to_expiry) * stats.norm.cdf(
                d1
            ) - strike * np.exp(-rate * time_to_expiry) * stats.norm.cdf(d2)
        else:
            price = strike * np.exp(-rate * time_to_expiry) * stats.norm.cdf(
                -d2
            ) - spot * np.exp(-dividend * time_to_expiry) * stats.norm.cdf(-d1)
        return price
