"""
Volatility Surface construction and interpolation.

This module implements volatility surface construction and interpolation
for more accurate option pricing across different strikes and maturities.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VolatilitySurface:
    """
    Implementation of volatility surface construction and interpolation.
    
    This class provides methods to fit a volatility surface from market option data
    and interpolate volatilities for arbitrary strikes and maturities.
    """
    
    def __init__(self, config=None):
        """
        Initialize volatility surface.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.surface = None
        self.strike_grid = None
        self.time_grid = None
        self.method = self.config.get("method", "spline")
    
    def fit_surface(self, option_data):
        """
        Fit volatility surface from market option data.
        
        Args:
            option_data (dict or list): Option data
                If dict: Expected format is {"calls": [...], "puts": [...]}
                If list: List of option data dictionaries
                Each option should have strike, expiry, and iv (implied volatility)
            
        Returns:
            tuple: (surface, strike_grid, time_grid)
        """
        # Extract option data
        options = []
        
        if isinstance(option_data, dict):
            # Extract from dict format
            if "calls" in option_data:
                options.extend(option_data["calls"])
            if "puts" in option_data:
                options.extend(option_data["puts"])
        else:
            # Assume list format
            options = option_data
        
        # Extract unique strikes and expiries
        strikes = sorted(list(set([option["strike"] for option in options])))
        
        # Convert expiries to time to expiry in years if needed
        expiries = []
        for option in options:
            if isinstance(option["expiry"], str):
                # Convert date string to time to expiry
                expiry_date = datetime.strptime(option["expiry"], "%Y-%m-%d")
                today = datetime.now()
                time_to_expiry = (expiry_date - today).days / 365.0
                expiries.append(time_to_expiry)
            else:
                expiries.append(option["expiry"])
        
        expiries = sorted(list(set(expiries)))
        
        # Ensure we have at least 4 unique strikes and expiries for spline interpolation
        if len(strikes) < 4:
            # Add synthetic strikes
            min_strike = min(strikes)
            max_strike = max(strikes)
            step = (max_strike - min_strike) / 10
            strikes = list(np.linspace(min_strike - step, max_strike + step, 12))
        
        if len(expiries) < 4:
            # Add synthetic expiries
            min_expiry = min(expiries)
            max_expiry = max(expiries)
            if max_expiry <= min_expiry:
                max_expiry = min_expiry + 1.0
            expiries = list(np.linspace(max(0.01, min_expiry - 0.1), max_expiry + 0.1, 12))
        
        # Create grids
        strike_grid = np.array(strikes)
        time_grid = np.array(expiries)
        
        # Create implied volatility surface
        surface = np.zeros((len(time_grid), len(strike_grid)))
        
        # Fill in known implied volatilities
        for option in options:
            strike = option["strike"]
            
            # Get time to expiry
            if isinstance(option["expiry"], str):
                expiry_date = datetime.strptime(option["expiry"], "%Y-%m-%d")
                today = datetime.now()
                time_to_expiry = (expiry_date - today).days / 365.0
            else:
                time_to_expiry = option["expiry"]
            
            # Get implied volatility
            iv = option.get("iv")
            
            if iv is None:
                # Skip if no implied volatility
                continue
            
            # Find closest grid points
            i = np.abs(time_grid - time_to_expiry).argmin()
            j = np.abs(strike_grid - strike).argmin()
            
            # Set implied volatility
            surface[i, j] = iv
        
        # Fill in missing values using interpolation
        mask = surface == 0
        if np.any(mask):
            # Create grid points
            xx, yy = np.meshgrid(strike_grid, time_grid)
            # Get non-zero points
            points = np.vstack((xx[~mask].ravel(), yy[~mask].ravel())).T
            values = surface[~mask].ravel()
            
            # If we don't have enough points for interpolation, add synthetic points
            if len(points) < 4:
                # Add synthetic points at corners with reasonable volatility values
                synthetic_points = []
                synthetic_values = []
                
                # Use average volatility as baseline
                avg_vol = np.mean(values) if len(values) > 0 else 0.2
                
                # Add corner points
                synthetic_points.append([strike_grid[0], time_grid[0]])
                synthetic_values.append(avg_vol * 1.2)  # Higher vol for short-term OTM
                
                synthetic_points.append([strike_grid[-1], time_grid[0]])
                synthetic_values.append(avg_vol * 1.2)  # Higher vol for short-term OTM
                
                synthetic_points.append([strike_grid[0], time_grid[-1]])
                synthetic_values.append(avg_vol * 1.1)  # Slightly higher vol for long-term OTM
                
                synthetic_points.append([strike_grid[-1], time_grid[-1]])
                synthetic_values.append(avg_vol * 1.1)  # Slightly higher vol for long-term OTM
                
                # Add ATM points
                mid_strike_idx = len(strike_grid) // 2
                synthetic_points.append([strike_grid[mid_strike_idx], time_grid[0]])
                synthetic_values.append(avg_vol * 0.9)  # Lower vol for ATM
                
                synthetic_points.append([strike_grid[mid_strike_idx], time_grid[-1]])
                synthetic_values.append(avg_vol * 0.95)  # Slightly lower vol for long-term ATM
                
                # Add to existing points
                points = np.vstack((points, synthetic_points)) if len(points) > 0 else np.array(synthetic_points)
                values = np.append(values, synthetic_values) if len(values) > 0 else np.array(synthetic_values)
            
            # Interpolate
            if self.method == "spline" and len(points) >= 16:  # Need enough points for spline
                # Use spline interpolation
                try:
                    # Create temporary grid for spline
                    temp_strike_grid = np.linspace(strike_grid.min(), strike_grid.max(), 20)
                    temp_time_grid = np.linspace(time_grid.min(), time_grid.max(), 20)
                    temp_xx, temp_yy = np.meshgrid(temp_strike_grid, temp_time_grid)
                    
                    # Interpolate to temporary grid first
                    temp_surface = griddata(points, values, (temp_xx, temp_yy), method='cubic', fill_value=np.nan)
                    
                    # Fill NaNs with linear interpolation
                    mask = np.isnan(temp_surface)
                    if np.any(mask):
                        temp_surface[mask] = griddata(points, values, (temp_xx[mask], temp_yy[mask]), method='linear', fill_value=np.nan)
                    
                    # Fill remaining NaNs with nearest neighbor
                    mask = np.isnan(temp_surface)
                    if np.any(mask):
                        temp_surface[mask] = griddata(points, values, (temp_xx[mask], temp_yy[mask]), method='nearest')
                    
                    # Create spline from temporary grid
                    spline = RectBivariateSpline(temp_time_grid, temp_strike_grid, temp_surface)
                    
                    # Evaluate spline on original grid
                    for i, t in enumerate(time_grid):
                        for j, k in enumerate(strike_grid):
                            if surface[i, j] == 0:
                                surface[i, j] = spline(t, k)[0, 0]
                except Exception as e:
                    logger.warning(f"Spline interpolation failed: {str(e)}. Falling back to griddata.")
                    # Fall back to griddata
                    surface = griddata(points, values, (xx, yy), method='cubic', fill_value=np.nan)
                    
                    # Fill NaNs with linear interpolation
                    mask = np.isnan(surface)
                    if np.any(mask):
                        surface[mask] = griddata(points, values, (xx[mask], yy[mask]), method='linear', fill_value=np.nan)
                    
                    # Fill remaining NaNs with nearest neighbor
                    mask = np.isnan(surface)
                    if np.any(mask):
                        surface[mask] = griddata(points, values, (xx[mask], yy[mask]), method='nearest')
            else:
                # Use griddata interpolation
                surface = griddata(points, values, (xx, yy), method='cubic', fill_value=np.nan)
                
                # Fill NaNs with linear interpolation
                mask = np.isnan(surface)
                if np.any(mask):
                    surface[mask] = griddata(points, values, (xx[mask], yy[mask]), method='linear', fill_value=np.nan)
                
                # Fill remaining NaNs with nearest neighbor
                mask = np.isnan(surface)
                if np.any(mask):
                    surface[mask] = griddata(points, values, (xx[mask], yy[mask]), method='nearest')
        
        # Apply smoothing if configured
        if self.config.get("apply_smoothing", True):
            surface = self._smooth_surface(surface)
        
        # Apply arbitrage-free adjustments if configured
        if self.config.get("arbitrage_free", True):
            surface = self._ensure_arbitrage_free(surface, strike_grid, time_grid)
        
        # Store results
        self.surface = surface
        self.strike_grid = strike_grid
        self.time_grid = time_grid
        
        return surface, strike_grid, time_grid
    
    def get_volatility(self, strike, time_to_expiry):
        """
        Get implied volatility for a specific strike and time to expiry.
        
        Args:
            strike (float): Strike price
            time_to_expiry (float): Time to expiry in years
            
        Returns:
            float: Implied volatility
        """
        if self.surface is None:
            raise ValueError("Volatility surface not fitted")
        
        # Ensure strike and time_to_expiry are within bounds
        strike = max(min(strike, self.strike_grid[-1]), self.strike_grid[0])
        time_to_expiry = max(min(time_to_expiry, self.time_grid[-1]), self.time_grid[0])
        
        # Interpolate
        if self.method == "spline":
            try:
                spline = RectBivariateSpline(self.time_grid, self.strike_grid, self.surface)
                return float(spline(time_to_expiry, strike)[0, 0])
            except Exception as e:
                logger.warning(f"Spline interpolation failed: {str(e)}. Falling back to linear interpolation.")
                # Fall back to linear interpolation
                return self._interpolate_linear(strike, time_to_expiry)
        else:
            return self._interpolate_linear(strike, time_to_expiry)
    
    def _interpolate_linear(self, strike, time_to_expiry):
        """
        Perform linear interpolation.
        
        Args:
            strike (float): Strike price
            time_to_expiry (float): Time to expiry in years
            
        Returns:
            float: Interpolated implied volatility
        """
        # Find nearest grid points
        i_time = np.searchsorted(self.time_grid, time_to_expiry)
        if i_time == 0:
            i_time = 1
        elif i_time == len(self.time_grid):
            i_time = len(self.time_grid) - 1
        
        i_strike = np.searchsorted(self.strike_grid, strike)
        if i_strike == 0:
            i_strike = 1
        elif i_strike == len(self.strike_grid):
            i_strike = len(self.strike_grid) - 1
        
        # Get surrounding grid points
        t1, t2 = self.time_grid[i_time-1], self.time_grid[i_time]
        k1, k2 = self.strike_grid[i_strike-1], self.strike_grid[i_strike]
        
        # Get volatilities at surrounding grid points
        v11 = self.surface[i_time-1, i_strike-1]
        v12 = self.surface[i_time-1, i_strike]
        v21 = self.surface[i_time, i_strike-1]
        v22 = self.surface[i_time, i_strike]
        
        # Perform bilinear interpolation
        dt = (time_to_expiry - t1) / (t2 - t1)
        dk = (strike - k1) / (k2 - k1)
        
        v1 = v11 * (1 - dk) + v12 * dk
        v2 = v21 * (1 - dk) + v22 * dk
        
        return v1 * (1 - dt) + v2 * dt
    
    def _smooth_surface(self, surface):
        """
        Apply smoothing to the volatility surface.
        
        Args:
            surface (array): Volatility surface
            
        Returns:
            array: Smoothed volatility surface
        """
        # Simple smoothing using convolution
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1, 0.4, 0.1],
                           [0.05, 0.1, 0.05]])
        
        # Pad surface to handle edges
        padded = np.pad(surface, ((1, 1), (1, 1)), mode='edge')
        
        # Apply convolution
        smoothed = np.zeros_like(surface)
        for i in range(surface.shape[0]):
            for j in range(surface.shape[1]):
                smoothed[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        
        return smoothed
    
    def _ensure_arbitrage_free(self, surface, strike_grid, time_grid):
        """
        Ensure the volatility surface is arbitrage-free.
        
        Args:
            surface (array): Volatility surface
            strike_grid (array): Strike price grid
            time_grid (array): Time to expiry grid
            
        Returns:
            array: Arbitrage-free volatility surface
        """
        # Check and fix calendar arbitrage (volatility should increase with time)
        for j in range(surface.shape[1]):
            for i in range(1, surface.shape[0]):
                if surface[i, j] < surface[i-1, j]:
                    # Apply a small increment to ensure monotonicity
                    surface[i, j] = surface[i-1, j] + 0.0001
        
        # Check and fix butterfly arbitrage (convexity in strike dimension)
        for i in range(surface.shape[0]):
            for j in range(1, surface.shape[1]-1):
                # Calculate second derivative
                d2v = (surface[i, j+1] - 2*surface[i, j] + surface[i, j-1])
                
                if d2v < 0:
                    # Adjust to ensure convexity
                    surface[i, j] = (surface[i, j+1] + surface[i, j-1]) / 2 - 0.0001
        
        return surface
    
    def plot_surface(self, title=None):
        """
        Plot the volatility surface.
        
        Args:
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.surface is None:
            raise ValueError("Volatility surface not fitted")
        
        # Create meshgrid
        X, Y = np.meshgrid(self.strike_grid, self.time_grid)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X, Y, self.surface, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry')
        ax.set_zlabel('Implied Volatility')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Implied Volatility Surface')
        
        return fig
    
    def plot_smile(self, time_to_expiry, title=None):
        """
        Plot the volatility smile for a specific time to expiry.
        
        Args:
            time_to_expiry (float): Time to expiry in years
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.surface is None:
            raise ValueError("Volatility surface not fitted")
        
        # Find closest time index
        time_idx = np.abs(self.time_grid - time_to_expiry).argmin()
        actual_time = self.time_grid[time_idx]
        
        # Extract volatility smile
        smile = self.surface[time_idx, :]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.strike_grid, smile, 'b-', linewidth=2)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Implied Volatility')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Volatility Smile (T={actual_time:.2f})')
        
        ax.grid(True)
        
        return fig
    
    def plot_term_structure(self, strike, title=None):
        """
        Plot the volatility term structure for a specific strike.
        
        Args:
            strike (float): Strike price
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.surface is None:
            raise ValueError("Volatility surface not fitted")
        
        # Find closest strike index
        strike_idx = np.abs(self.strike_grid - strike).argmin()
        actual_strike = self.strike_grid[strike_idx]
        
        # Extract term structure
        term_structure = self.surface[:, strike_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time_grid, term_structure, 'r-', linewidth=2)
        ax.set_xlabel('Time to Expiry')
        ax.set_ylabel('Implied Volatility')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Volatility Term Structure (K={actual_strike:.2f})')
        
        ax.grid(True)
        
        return fig
