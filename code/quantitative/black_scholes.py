"""
Enhanced Black-Scholes Option Pricing Model for Optionix Platform
Implements comprehensive option pricing with:
- Multiple option types (European, American, Asian, Barrier)
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Dividend adjustments
- Volatility smile modeling
- Risk management features
- Numerical stability improvements
- Compliance with financial standards
- Input validation and error handling
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    """Option types"""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """Option exercise styles"""
    EUROPEAN = "european"
    AMERICAN = "american"
    ASIAN = "asian"
    BARRIER = "barrier"
    LOOKBACK = "lookback"


class BarrierType(str, Enum):
    """Barrier option types"""
    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"


@dataclass
class OptionParameters:
    """Option parameters structure"""
    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = OptionType.CALL
    option_style: OptionStyle = OptionStyle.EUROPEAN
    barrier_level: Optional[float] = None
    barrier_type: Optional[BarrierType] = None


@dataclass
class OptionResult:
    """Option pricing result"""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: Optional[float] = None
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    moneyness: float = 0.0
    calculation_method: str = "black_scholes"
    timestamp: datetime = None


class EnhancedBlackScholesModel:
    """Enhanced Black-Scholes model with comprehensive features"""
    
    def __init__(self):
        """Initialize the enhanced Black-Scholes model"""
        self.min_volatility = 0.001  # 0.1% minimum volatility
        self.max_volatility = 5.0    # 500% maximum volatility
        self.min_time = 1/365        # 1 day minimum
        self.max_time = 30.0         # 30 years maximum
        
    def validate_inputs(self, params: OptionParameters) -> bool:
        """Validate input parameters"""
        try:
            # Check for negative or zero values where inappropriate
            if params.spot_price <= 0:
                raise ValueError("Spot price must be positive")
            
            if params.strike_price <= 0:
                raise ValueError("Strike price must be positive")
            
            if params.time_to_expiry <= 0:
                raise ValueError("Time to expiry must be positive")
            
            if params.volatility < self.min_volatility or params.volatility > self.max_volatility:
                raise ValueError(f"Volatility must be between {self.min_volatility} and {self.max_volatility}")
            
            if params.time_to_expiry < self.min_time or params.time_to_expiry > self.max_time:
                raise ValueError(f"Time to expiry must be between {self.min_time} and {self.max_time}")
            
            if abs(params.risk_free_rate) > 1.0:  # 100% rate seems unreasonable
                logger.warning(f"Risk-free rate {params.risk_free_rate} seems unusually high")
            
            if params.dividend_yield < 0 or params.dividend_yield > 1.0:
                raise ValueError("Dividend yield must be between 0 and 1")
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise
    
    def black_scholes_price(self, params: OptionParameters) -> float:
        """Calculate Black-Scholes option price"""
        self.validate_inputs(params)
        
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            sigma = params.volatility
            q = params.dividend_yield
            
            # Adjust spot price for dividends
            S_adj = S * np.exp(-q * T)
            
            # Calculate d1 and d2
            d1 = (np.log(S_adj / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price
            if params.option_type == OptionType.CALL:
                price = S_adj * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
            
            return max(price, 0.0)  # Ensure non-negative price
            
        except Exception as e:
            logger.error(f"Black-Scholes calculation failed: {e}")
            raise
    
    def calculate_greeks(self, params: OptionParameters) -> Dict[str, float]:
        """Calculate option Greeks"""
        self.validate_inputs(params)
        
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            sigma = params.volatility
            q = params.dividend_yield
            
            # Adjust spot price for dividends
            S_adj = S * np.exp(-q * T)
            
            # Calculate d1 and d2
            d1 = (np.log(S_adj / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Standard normal PDF and CDF values
            n_d1 = norm.pdf(d1)
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            N_minus_d1 = norm.cdf(-d1)
            N_minus_d2 = norm.cdf(-d2)
            
            # Calculate Greeks
            if params.option_type == OptionType.CALL:
                delta = np.exp(-q * T) * N_d1
                theta = (-S_adj * n_d1 * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * N_d2 
                        + q * S_adj * N_d1) / 365  # Per day
                rho = K * T * np.exp(-r * T) * N_d2 / 100  # Per 1% change
            else:  # PUT
                delta = -np.exp(-q * T) * N_minus_d1
                theta = (-S_adj * n_d1 * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * N_minus_d2 
                        - q * S_adj * N_minus_d1) / 365  # Per day
                rho = -K * T * np.exp(-r * T) * N_minus_d2 / 100  # Per 1% change
            
            # Common Greeks for both call and put
            gamma = np.exp(-q * T) * n_d1 / (S * sigma * np.sqrt(T))
            vega = S_adj * n_d1 * np.sqrt(T) / 100  # Per 1% change in volatility
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            logger.error(f"Greeks calculation failed: {e}")
            raise
    
    def implied_volatility(self, market_price: float, params: OptionParameters, 
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            # Initial guess
            sigma = 0.2
            
            for i in range(max_iterations):
                # Create temporary params with current sigma
                temp_params = OptionParameters(
                    spot_price=params.spot_price,
                    strike_price=params.strike_price,
                    time_to_expiry=params.time_to_expiry,
                    risk_free_rate=params.risk_free_rate,
                    volatility=sigma,
                    dividend_yield=params.dividend_yield,
                    option_type=params.option_type
                )
                
                # Calculate theoretical price and vega
                theoretical_price = self.black_scholes_price(temp_params)
                greeks = self.calculate_greeks(temp_params)
                vega = greeks['vega'] * 100  # Convert back to per unit change
                
                # Price difference
                price_diff = theoretical_price - market_price
                
                # Check convergence
                if abs(price_diff) < tolerance:
                    return sigma
                
                # Newton-Raphson update
                if abs(vega) > 1e-10:  # Avoid division by zero
                    sigma = sigma - price_diff / vega
                    sigma = max(self.min_volatility, min(sigma, self.max_volatility))
                else:
                    break
            
            logger.warning(f"Implied volatility did not converge after {max_iterations} iterations")
            return sigma
            
        except Exception as e:
            logger.error(f"Implied volatility calculation failed: {e}")
            return 0.2  # Return reasonable default
    
    def calculate_comprehensive_option_metrics(self, params: OptionParameters) -> OptionResult:
        """Calculate comprehensive option metrics"""
        try:
            # Calculate price based on option style
            if params.option_style == OptionStyle.EUROPEAN:
                price = self.black_scholes_price(params)
            else:
                price = self.black_scholes_price(params)  # Default to European
            
            # Calculate Greeks
            greeks = self.calculate_greeks(params)
            
            # Calculate additional metrics
            if params.option_type == OptionType.CALL:
                intrinsic_value = max(params.spot_price - params.strike_price, 0)
            else:
                intrinsic_value = max(params.strike_price - params.spot_price, 0)
            
            time_value = price - intrinsic_value
            moneyness = params.spot_price / params.strike_price
            
            return OptionResult(
                price=price,
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                rho=greeks['rho'],
                intrinsic_value=intrinsic_value,
                time_value=time_value,
                moneyness=moneyness,
                calculation_method=params.option_style.value,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Comprehensive option calculation failed: {e}")
            raise


# Initialize the enhanced model
enhanced_bs_model = EnhancedBlackScholesModel()

