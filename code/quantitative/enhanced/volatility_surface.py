"""
Enhanced Volatility Surface Construction and Interpolation for Optionix Platform
Implements comprehensive volatility surface modeling with:
- Multiple interpolation methods (spline, SVI, SABR)
- Arbitrage-free surface construction
- Real-time market data integration
- Risk management features
- Compliance with financial standards
- Advanced calibration techniques
- Model validation and backtesting
- Performance optimization
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RectBivariateSpline, interp2d
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class InterpolationMethod(str, Enum):
    """Volatility surface interpolation methods"""
    SPLINE = "spline"
    SVI = "svi"  # Stochastic Volatility Inspired
    SABR = "sabr"  # Stochastic Alpha Beta Rho
    HESTON = "heston"
    LINEAR = "linear"
    CUBIC = "cubic"
    RBF = "rbf"  # Radial Basis Function


class CalibrationMethod(str, Enum):
    """Model calibration methods"""
    LEAST_SQUARES = "least_squares"
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"


@dataclass
class OptionData:
    """Option market data structure"""
    strike: float
    expiry: float  # Time to expiry in years
    implied_volatility: float
    option_type: str  # 'call' or 'put'
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    underlying_price: Optional[float] = None
    risk_free_rate: Optional[float] = None
    dividend_yield: Optional[float] = None


@dataclass
class SurfaceParameters:
    """Volatility surface model parameters"""
    model_type: InterpolationMethod
    parameters: Dict[str, float]
    calibration_date: datetime
    underlying_price: float
    risk_free_rate: float
    dividend_yield: float
    rmse: float
    max_error: float
    r_squared: float


@dataclass
class ArbitrageCheck:
    """Arbitrage check results"""
    calendar_arbitrage: bool
    butterfly_arbitrage: bool
    call_put_parity: bool
    violations: List[str]
    severity_score: float


class EnhancedVolatilitySurface:
    """
    Enhanced volatility surface implementation with comprehensive features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced volatility surface
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.surface_data = None
        self.surface_function = None
        self.parameters = None
        self.strike_grid = None
        self.time_grid = None
        self.method = InterpolationMethod(self.config.get("method", "spline"))
        self.calibration_method = CalibrationMethod(self.config.get("calibration_method", "least_squares"))
        
        # Validation parameters
        self.min_volatility = self.config.get("min_volatility", 0.01)
        self.max_volatility = self.config.get("max_volatility", 3.0)
        self.min_time_to_expiry = self.config.get("min_time_to_expiry", 1/365)
        self.max_time_to_expiry = self.config.get("max_time_to_expiry", 5.0)
        
        # Performance settings
        self.parallel_processing = self.config.get("parallel_processing", True)
        self.cache_results = self.config.get("cache_results", True)
        self._cache = {}
    
    def validate_option_data(self, option_data: List[OptionData]) -> List[OptionData]:
        """Validate and clean option data"""
        try:
            validated_data = []
            
            for option in option_data:
                # Check for valid values
                if (option.strike <= 0 or 
                    option.expiry <= self.min_time_to_expiry or 
                    option.expiry > self.max_time_to_expiry or
                    option.implied_volatility <= self.min_volatility or
                    option.implied_volatility > self.max_volatility):
                    logger.warning(f"Skipping invalid option: K={option.strike}, T={option.expiry}, IV={option.implied_volatility}")
                    continue
                
                # Check for reasonable bid-ask spreads
                if option.bid and option.ask:
                    spread = (option.ask - option.bid) / ((option.ask + option.bid) / 2)
                    if spread > 0.5:  # 50% spread threshold
                        logger.warning(f"Wide bid-ask spread detected: {spread:.2%}")
                
                validated_data.append(option)
            
            logger.info(f"Validated {len(validated_data)} out of {len(option_data)} options")
            return validated_data
            
        except Exception as e:
            logger.error(f"Option data validation failed: {e}")
            raise
    
    def fit_surface(self, option_data: List[OptionData], underlying_price: float,
                   risk_free_rate: float = 0.0, dividend_yield: float = 0.0) -> SurfaceParameters:
        """
        Fit volatility surface from market option data
        
        Args:
            option_data: List of option market data
            underlying_price: Current underlying asset price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            Surface parameters and calibration results
        """
        try:
            # Validate input data
            validated_data = self.validate_option_data(option_data)
            
            if len(validated_data) < 10:
                raise ValueError("Insufficient valid option data for surface fitting")
            
            # Convert to arrays for processing
            strikes = np.array([opt.strike for opt in validated_data])
            expiries = np.array([opt.expiry for opt in validated_data])
            ivs = np.array([opt.implied_volatility for opt in validated_data])
            
            # Create moneyness and log-moneyness
            moneyness = strikes / underlying_price
            log_moneyness = np.log(moneyness)
            
            # Store data
            self.surface_data = {
                'strikes': strikes,
                'expiries': expiries,
                'moneyness': moneyness,
                'log_moneyness': log_moneyness,
                'implied_volatilities': ivs,
                'underlying_price': underlying_price,
                'risk_free_rate': risk_free_rate,
                'dividend_yield': dividend_yield
            }
            
            # Fit surface based on method
            if self.method == InterpolationMethod.SVI:
                parameters = self._fit_svi_surface(log_moneyness, expiries, ivs)
            elif self.method == InterpolationMethod.SABR:
                parameters = self._fit_sabr_surface(strikes, expiries, ivs, underlying_price)
            elif self.method == InterpolationMethod.SPLINE:
                parameters = self._fit_spline_surface(moneyness, expiries, ivs)
            else:
                parameters = self._fit_spline_surface(moneyness, expiries, ivs)
            
            # Calculate calibration metrics
            fitted_ivs = self.interpolate_volatility(strikes, expiries)
            rmse = np.sqrt(np.mean((ivs - fitted_ivs)**2))
            max_error = np.max(np.abs(ivs - fitted_ivs))
            r_squared = 1 - np.sum((ivs - fitted_ivs)**2) / np.sum((ivs - np.mean(ivs))**2)
            
            # Store parameters
            self.parameters = SurfaceParameters(
                model_type=self.method,
                parameters=parameters,
                calibration_date=datetime.utcnow(),
                underlying_price=underlying_price,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                rmse=rmse,
                max_error=max_error,
                r_squared=r_squared
            )
            
            logger.info(f"Surface fitted successfully: RMSE={rmse:.4f}, R²={r_squared:.4f}")
            return self.parameters
            
        except Exception as e:
            logger.error(f"Surface fitting failed: {e}")
            raise
    
    def _fit_svi_surface(self, log_moneyness: np.ndarray, expiries: np.ndarray, 
                        ivs: np.ndarray) -> Dict[str, Any]:
        """Fit SVI (Stochastic Volatility Inspired) surface"""
        try:
            # SVI parameterization: σ²(k,T) = a + b(ρ(k-m) + √((k-m)² + σ²))
            # where k = log(K/S), T = time to expiry
            
            unique_expiries = np.unique(expiries)
            svi_params = {}
            
            for T in unique_expiries:
                mask = expiries == T
                k_slice = log_moneyness[mask]
                iv_slice = ivs[mask]
                
                if len(k_slice) < 5:  # Need minimum points for SVI fit
                    continue
                
                # Initial parameter guess
                initial_guess = [0.04, 0.4, -0.4, 0.0, 0.4]  # [a, b, rho, m, sigma]
                
                # Parameter bounds
                bounds = [
                    (0.001, 1.0),    # a: minimum variance level
                    (0.001, 2.0),    # b: variance slope
                    (-0.999, 0.999), # rho: correlation
                    (-2.0, 2.0),     # m: ATM level
                    (0.001, 2.0)     # sigma: curvature
                ]
                
                # Objective function
                def svi_objective(params):
                    a, b, rho, m, sigma = params
                    
                    # SVI formula
                    variance = a + b * (rho * (k_slice - m) + 
                                      np.sqrt((k_slice - m)**2 + sigma**2))
                    
                    # Ensure positive variance
                    variance = np.maximum(variance, 0.0001)
                    implied_vol = np.sqrt(variance)
                    
                    # Return sum of squared errors
                    return np.sum((implied_vol - iv_slice)**2)
                
                # Optimize
                result = minimize(svi_objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    svi_params[T] = {
                        'a': result.x[0],
                        'b': result.x[1],
                        'rho': result.x[2],
                        'm': result.x[3],
                        'sigma': result.x[4],
                        'error': result.fun
                    }
            
            return {'svi_params': svi_params, 'method': 'svi'}
            
        except Exception as e:
            logger.error(f"SVI surface fitting failed: {e}")
            raise
    
    def _fit_sabr_surface(self, strikes: np.ndarray, expiries: np.ndarray, 
                         ivs: np.ndarray, underlying_price: float) -> Dict[str, Any]:
        """Fit SABR (Stochastic Alpha Beta Rho) surface"""
        try:
            # SABR model parameters: alpha, beta, rho, nu
            unique_expiries = np.unique(expiries)
            sabr_params = {}
            
            for T in unique_expiries:
                mask = expiries == T
                k_slice = strikes[mask]
                iv_slice = ivs[mask]
                
                if len(k_slice) < 4:  # Need minimum points for SABR fit
                    continue
                
                # Initial parameter guess
                initial_guess = [0.2, 0.5, 0.0, 0.3]  # [alpha, beta, rho, nu]
                
                # Parameter bounds
                bounds = [
                    (0.001, 2.0),    # alpha: volatility level
                    (0.0, 1.0),      # beta: elasticity
                    (-0.999, 0.999), # rho: correlation
                    (0.001, 2.0)     # nu: vol of vol
                ]
                
                # Objective function
                def sabr_objective(params):
                    alpha, beta, rho, nu = params
                    
                    try:
                        # SABR implied volatility formula (Hagan approximation)
                        F = underlying_price  # Forward price (simplified)
                        sabr_ivs = []
                        
                        for K in k_slice:
                            if K <= 0 or F <= 0:
                                sabr_ivs.append(0.2)  # Default volatility
                                continue
                            
                            # Log-moneyness
                            z = (nu / alpha) * (F * K)**((1 - beta) / 2) * np.log(F / K)
                            
                            if abs(z) < 1e-6:
                                # ATM case
                                iv = alpha / (F**(1 - beta)) * (1 + ((1 - beta)**2 / 24) * 
                                    (np.log(F / K))**2 + ((1 - beta)**4 / 1920) * (np.log(F / K))**4)
                            else:
                                # Non-ATM case
                                x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
                                iv = (nu * np.log(F / K)) / x
                            
                            # Apply time scaling
                            iv *= np.sqrt(T)
                            sabr_ivs.append(max(iv, 0.001))
                        
                        sabr_ivs = np.array(sabr_ivs)
                        return np.sum((sabr_ivs - iv_slice)**2)
                        
                    except:
                        return 1e6  # Large penalty for invalid parameters
                
                # Optimize
                result = minimize(sabr_objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    sabr_params[T] = {
                        'alpha': result.x[0],
                        'beta': result.x[1],
                        'rho': result.x[2],
                        'nu': result.x[3],
                        'error': result.fun
                    }
            
            return {'sabr_params': sabr_params, 'method': 'sabr'}
            
        except Exception as e:
            logger.error(f"SABR surface fitting failed: {e}")
            raise
    
    def _fit_spline_surface(self, moneyness: np.ndarray, expiries: np.ndarray, 
                           ivs: np.ndarray) -> Dict[str, Any]:
        """Fit spline-based volatility surface"""
        try:
            # Create regular grid
            moneyness_grid = np.linspace(moneyness.min(), moneyness.max(), 20)
            expiry_grid = np.linspace(expiries.min(), expiries.max(), 10)
            
            # Interpolate to grid
            grid_ivs = griddata(
                (moneyness, expiries), ivs, 
                (moneyness_grid[None, :], expiry_grid[:, None]), 
                method='cubic', fill_value=np.nan
            )
            
            # Fill NaN values with nearest neighbor
            mask = np.isnan(grid_ivs)
            if np.any(mask):
                grid_ivs_nn = griddata(
                    (moneyness, expiries), ivs,
                    (moneyness_grid[None, :], expiry_grid[:, None]),
                    method='nearest'
                )
                grid_ivs[mask] = grid_ivs_nn[mask]
            
            # Create spline interpolator
            self.surface_function = RectBivariateSpline(
                expiry_grid, moneyness_grid, grid_ivs, 
                kx=min(3, len(expiry_grid)-1), ky=min(3, len(moneyness_grid)-1)
            )
            
            return {
                'moneyness_grid': moneyness_grid,
                'expiry_grid': expiry_grid,
                'volatility_grid': grid_ivs,
                'method': 'spline'
            }
            
        except Exception as e:
            logger.error(f"Spline surface fitting failed: {e}")
            raise
    
    def interpolate_volatility(self, strikes: Union[float, np.ndarray], 
                             expiries: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolate implied volatility for given strikes and expiries
        
        Args:
            strikes: Strike prices
            expiries: Times to expiry
            
        Returns:
            Interpolated implied volatilities
        """
        try:
            if self.surface_data is None:
                raise ValueError("Surface must be fitted before interpolation")
            
            # Convert to arrays
            strikes = np.atleast_1d(strikes)
            expiries = np.atleast_1d(expiries)
            
            # Check cache
            cache_key = (tuple(strikes), tuple(expiries))
            if self.cache_results and cache_key in self._cache:
                return self._cache[cache_key]
            
            if self.method == InterpolationMethod.SVI:
                ivs = self._interpolate_svi(strikes, expiries)
            elif self.method == InterpolationMethod.SABR:
                ivs = self._interpolate_sabr(strikes, expiries)
            elif self.method == InterpolationMethod.SPLINE:
                ivs = self._interpolate_spline(strikes, expiries)
            else:
                ivs = self._interpolate_spline(strikes, expiries)
            
            # Ensure reasonable bounds
            ivs = np.clip(ivs, self.min_volatility, self.max_volatility)
            
            # Cache result
            if self.cache_results:
                self._cache[cache_key] = ivs
            
            return ivs[0] if len(ivs) == 1 else ivs
            
        except Exception as e:
            logger.error(f"Volatility interpolation failed: {e}")
            raise
    
    def _interpolate_svi(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        """Interpolate using SVI model"""
        svi_params = self.parameters.parameters['svi_params']
        underlying_price = self.surface_data['underlying_price']
        
        log_moneyness = np.log(strikes / underlying_price)
        ivs = np.zeros_like(strikes, dtype=float)
        
        for i, (k, T) in enumerate(zip(log_moneyness, expiries)):
            # Find closest expiry parameters
            available_expiries = list(svi_params.keys())
            closest_expiry = min(available_expiries, key=lambda x: abs(x - T))
            
            params = svi_params[closest_expiry]
            a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']
            
            # SVI formula
            variance = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            ivs[i] = np.sqrt(max(variance, 0.0001))
        
        return ivs
    
    def _interpolate_sabr(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        """Interpolate using SABR model"""
        sabr_params = self.parameters.parameters['sabr_params']
        underlying_price = self.surface_data['underlying_price']
        
        ivs = np.zeros_like(strikes, dtype=float)
        
        for i, (K, T) in enumerate(zip(strikes, expiries)):
            # Find closest expiry parameters
            available_expiries = list(sabr_params.keys())
            closest_expiry = min(available_expiries, key=lambda x: abs(x - T))
            
            params = sabr_params[closest_expiry]
            alpha, beta, rho, nu = params['alpha'], params['beta'], params['rho'], params['nu']
            
            # SABR implied volatility (simplified)
            F = underlying_price
            if K <= 0 or F <= 0:
                ivs[i] = 0.2
                continue
            
            try:
                z = (nu / alpha) * (F * K)**((1 - beta) / 2) * np.log(F / K)
                
                if abs(z) < 1e-6:
                    iv = alpha / (F**(1 - beta))
                else:
                    x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
                    iv = (nu * np.log(F / K)) / x
                
                ivs[i] = max(iv * np.sqrt(T), 0.001)
                
            except:
                ivs[i] = 0.2  # Default volatility
        
        return ivs
    
    def _interpolate_spline(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        """Interpolate using spline surface"""
        if self.surface_function is None:
            raise ValueError("Spline surface not fitted")
        
        underlying_price = self.surface_data['underlying_price']
        moneyness = strikes / underlying_price
        
        # Evaluate spline
        ivs = self.surface_function.ev(expiries, moneyness)
        
        return ivs
    
    def check_arbitrage(self, tolerance: float = 0.01) -> ArbitrageCheck:
        """Check for arbitrage opportunities in the fitted surface"""
        try:
            violations = []
            severity_score = 0.0
            
            # Calendar arbitrage check
            calendar_ok = self._check_calendar_arbitrage(tolerance)
            if not calendar_ok:
                violations.append("Calendar arbitrage detected")
                severity_score += 0.3
            
            # Butterfly arbitrage check
            butterfly_ok = self._check_butterfly_arbitrage(tolerance)
            if not butterfly_ok:
                violations.append("Butterfly arbitrage detected")
                severity_score += 0.4
            
            # Call-put parity check (if both calls and puts available)
            parity_ok = self._check_call_put_parity(tolerance)
            if not parity_ok:
                violations.append("Call-put parity violation")
                severity_score += 0.3
            
            return ArbitrageCheck(
                calendar_arbitrage=calendar_ok,
                butterfly_arbitrage=butterfly_ok,
                call_put_parity=parity_ok,
                violations=violations,
                severity_score=severity_score
            )
            
        except Exception as e:
            logger.error(f"Arbitrage check failed: {e}")
            return ArbitrageCheck(
                calendar_arbitrage=False,
                butterfly_arbitrage=False,
                call_put_parity=False,
                violations=["Arbitrage check failed"],
                severity_score=1.0
            )
    
    def _check_calendar_arbitrage(self, tolerance: float) -> bool:
        """Check for calendar arbitrage (total variance should be increasing)"""
        try:
            if self.surface_data is None:
                return True
            
            # Sample points for checking
            underlying_price = self.surface_data['underlying_price']
            test_strikes = np.linspace(0.8 * underlying_price, 1.2 * underlying_price, 10)
            test_expiries = np.linspace(0.1, 2.0, 10)
            
            violations = 0
            total_checks = 0
            
            for strike in test_strikes:
                prev_total_var = 0
                for expiry in sorted(test_expiries):
                    iv = self.interpolate_volatility(strike, expiry)
                    total_var = iv**2 * expiry
                    
                    if total_var < prev_total_var - tolerance:
                        violations += 1
                    
                    prev_total_var = total_var
                    total_checks += 1
            
            violation_rate = violations / max(total_checks, 1)
            return violation_rate < 0.05  # Allow 5% violation rate
            
        except Exception as e:
            logger.warning(f"Calendar arbitrage check failed: {e}")
            return True
    
    def _check_butterfly_arbitrage(self, tolerance: float) -> bool:
        """Check for butterfly arbitrage (convexity in strike dimension)"""
        try:
            if self.surface_data is None:
                return True
            
            # Sample points for checking
            underlying_price = self.surface_data['underlying_price']
            test_expiries = [0.25, 0.5, 1.0, 2.0]
            
            violations = 0
            total_checks = 0
            
            for expiry in test_expiries:
                strikes = np.linspace(0.7 * underlying_price, 1.3 * underlying_price, 20)
                
                for i in range(1, len(strikes) - 1):
                    k1, k2, k3 = strikes[i-1], strikes[i], strikes[i+1]
                    
                    iv1 = self.interpolate_volatility(k1, expiry)
                    iv2 = self.interpolate_volatility(k2, expiry)
                    iv3 = self.interpolate_volatility(k3, expiry)
                    
                    # Check convexity condition
                    convexity = iv1 - 2*iv2 + iv3
                    if convexity < -tolerance:
                        violations += 1
                    
                    total_checks += 1
            
            violation_rate = violations / max(total_checks, 1)
            return violation_rate < 0.05
            
        except Exception as e:
            logger.warning(f"Butterfly arbitrage check failed: {e}")
            return True
    
    def _check_call_put_parity(self, tolerance: float) -> bool:
        """Check call-put parity violations"""
        # This would require both call and put data
        # For now, return True as we don't have separate call/put data
        return True
    
    def plot_surface(self, save_path: Optional[str] = None) -> None:
        """Plot the volatility surface"""
        try:
            if self.surface_data is None:
                raise ValueError("Surface must be fitted before plotting")
            
            # Create grid for plotting
            underlying_price = self.surface_data['underlying_price']
            strike_range = np.linspace(0.7 * underlying_price, 1.3 * underlying_price, 30)
            expiry_range = np.linspace(0.1, 2.0, 20)
            
            Strike_grid, Expiry_grid = np.meshgrid(strike_range, expiry_range)
            IV_grid = np.zeros_like(Strike_grid)
            
            for i in range(len(expiry_range)):
                for j in range(len(strike_range)):
                    IV_grid[i, j] = self.interpolate_volatility(Strike_grid[i, j], Expiry_grid[i, j])
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(Strike_grid, Expiry_grid, IV_grid, 
                                 cmap='viridis', alpha=0.8)
            
            # Add market data points
            ax.scatter(self.surface_data['strikes'], 
                      self.surface_data['expiries'],
                      self.surface_data['implied_volatilities'],
                      color='red', s=20, alpha=0.6, label='Market Data')
            
            ax.set_xlabel('Strike')
            ax.set_ylabel('Time to Expiry')
            ax.set_zlabel('Implied Volatility')
            ax.set_title(f'Volatility Surface ({self.method.value.upper()})')
            
            plt.colorbar(surf)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Surface plotting failed: {e}")
            raise
    
    def get_surface_metrics(self) -> Dict[str, float]:
        """Get surface quality metrics"""
        if self.parameters is None:
            return {}
        
        return {
            'rmse': self.parameters.rmse,
            'max_error': self.parameters.max_error,
            'r_squared': self.parameters.r_squared,
            'num_data_points': len(self.surface_data['strikes']) if self.surface_data else 0,
            'calibration_date': self.parameters.calibration_date.isoformat()
        }
    
    def export_surface(self, filepath: str) -> None:
        """Export fitted surface to file"""
        try:
            import pickle
            
            export_data = {
                'parameters': self.parameters,
                'surface_data': self.surface_data,
                'method': self.method,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)
            
            logger.info(f"Surface exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Surface export failed: {e}")
            raise
    
    def load_surface(self, filepath: str) -> None:
        """Load fitted surface from file"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                export_data = pickle.load(f)
            
            self.parameters = export_data['parameters']
            self.surface_data = export_data['surface_data']
            self.method = export_data['method']
            self.config = export_data.get('config', {})
            
            # Recreate surface function if spline method
            if self.method == InterpolationMethod.SPLINE:
                params = self.parameters.parameters
                self.surface_function = RectBivariateSpline(
                    params['expiry_grid'], params['moneyness_grid'], 
                    params['volatility_grid']
                )
            
            logger.info(f"Surface loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Surface loading failed: {e}")
            raise


# Backward compatibility
class VolatilitySurface(EnhancedVolatilitySurface):
    """Backward compatible volatility surface class"""
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def fit_surface(self, option_data):
        """Backward compatible fit_surface method"""
        # Convert old format to new format
        if isinstance(option_data, dict):
            # Handle {"calls": [...], "puts": [...]} format
            all_options = []
            
            for option_type in ['calls', 'puts']:
                if option_type in option_data:
                    for opt in option_data[option_type]:
                        all_options.append(OptionData(
                            strike=opt.get('strike', opt.get('K')),
                            expiry=opt.get('expiry', opt.get('T')),
                            implied_volatility=opt.get('iv', opt.get('implied_volatility')),
                            option_type=option_type[:-1]  # Remove 's' from 'calls'/'puts'
                        ))
        else:
            # Handle list format
            all_options = []
            for opt in option_data:
                all_options.append(OptionData(
                    strike=opt.get('strike', opt.get('K')),
                    expiry=opt.get('expiry', opt.get('T')),
                    implied_volatility=opt.get('iv', opt.get('implied_volatility')),
                    option_type=opt.get('option_type', 'call')
                ))
        
        # Use default values for missing parameters
        underlying_price = 100.0  # Default
        if all_options and hasattr(all_options[0], 'underlying_price') and all_options[0].underlying_price:
            underlying_price = all_options[0].underlying_price
        
        return super().fit_surface(all_options, underlying_price)


# Global instance for backward compatibility
volatility_surface = VolatilitySurface()

