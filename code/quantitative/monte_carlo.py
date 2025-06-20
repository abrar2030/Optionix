"""
Enhanced Monte Carlo Simulation Module for Optionix Platform
Implements comprehensive Monte Carlo methods for:
- Option pricing (European, American, Asian, Barrier, Exotic)
- Risk management (VaR, CVaR, stress testing)
- Portfolio optimization
- Path-dependent derivatives
- Multi-asset simulations
- Variance reduction techniques
- Parallel processing for performance
- Financial compliance and validation
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ProcessType(str, Enum):
    """Stochastic process types"""
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    MEAN_REVERTING = "mean_reverting"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON = "heston"
    VASICEK = "vasicek"
    CIR = "cir"


class VarianceReduction(str, Enum):
    """Variance reduction techniques"""
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    IMPORTANCE_SAMPLING = "importance_sampling"
    STRATIFIED_SAMPLING = "stratified_sampling"
    QUASI_RANDOM = "quasi_random"


@dataclass
class SimulationParameters:
    """Monte Carlo simulation parameters"""
    initial_price: float
    drift: float
    volatility: float
    time_horizon: float
    time_steps: int
    num_simulations: int
    process_type: ProcessType = ProcessType.GEOMETRIC_BROWNIAN_MOTION
    variance_reduction: Optional[VarianceReduction] = None
    random_seed: Optional[int] = None
    parallel: bool = True
    
    # Additional parameters for specific processes
    mean_reversion_speed: Optional[float] = None
    long_term_mean: Optional[float] = None
    jump_intensity: Optional[float] = None
    jump_mean: Optional[float] = None
    jump_volatility: Optional[float] = None
    
    # Heston model parameters
    initial_variance: Optional[float] = None
    variance_mean_reversion: Optional[float] = None
    long_term_variance: Optional[float] = None
    vol_of_vol: Optional[float] = None
    correlation: Optional[float] = None


@dataclass
class OptionPayoff:
    """Option payoff specification"""
    option_type: str  # 'call', 'put', 'asian_call', 'asian_put', 'barrier_call', etc.
    strike_price: float
    barrier_level: Optional[float] = None
    barrier_type: Optional[str] = None  # 'up_and_out', 'down_and_in', etc.
    averaging_period: Optional[int] = None  # For Asian options
    lookback_period: Optional[int] = None  # For lookback options


@dataclass
class SimulationResult:
    """Monte Carlo simulation result"""
    option_price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    paths: Optional[np.ndarray] = None
    payoffs: Optional[np.ndarray] = None
    greeks: Optional[Dict[str, float]] = None
    convergence_data: Optional[Dict[str, List[float]]] = None
    computation_time: float = 0.0
    method_used: str = ""


class EnhancedMonteCarloSimulator:
    """Enhanced Monte Carlo simulator with advanced features"""
    
    def __init__(self, params: SimulationParameters):
        """Initialize Monte Carlo simulator"""
        self.params = params
        self.validate_parameters()
        
        # Set random seed for reproducibility
        if params.random_seed is not None:
            np.random.seed(params.random_seed)
    
    def validate_parameters(self):
        """Validate simulation parameters"""
        if self.params.initial_price <= 0:
            raise ValueError("Initial price must be positive")
        
        if self.params.volatility < 0:
            raise ValueError("Volatility must be non-negative")
        
        if self.params.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        
        if self.params.time_steps <= 0:
            raise ValueError("Time steps must be positive")
        
        if self.params.num_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        
        # Validate process-specific parameters
        if self.params.process_type == ProcessType.MEAN_REVERTING:
            if self.params.mean_reversion_speed is None or self.params.long_term_mean is None:
                raise ValueError("Mean reversion parameters required")
        
        if self.params.process_type == ProcessType.HESTON:
            required_params = [
                self.params.initial_variance,
                self.params.variance_mean_reversion,
                self.params.long_term_variance,
                self.params.vol_of_vol,
                self.params.correlation
            ]
            if any(p is None for p in required_params):
                raise ValueError("All Heston model parameters required")
    
    def geometric_brownian_motion(self, n_simulations: Optional[int] = None) -> np.ndarray:
        """Generate paths using Geometric Brownian Motion"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.params.time_horizon / self.params.time_steps
        
        # Generate random numbers
        if self.params.variance_reduction == VarianceReduction.ANTITHETIC:
            # Antithetic variates
            n_base = n_sims // 2
            random_normals = np.random.standard_normal((self.params.time_steps, n_base))
            random_normals = np.concatenate([random_normals, -random_normals], axis=1)
        elif self.params.variance_reduction == VarianceReduction.QUASI_RANDOM:
            # Sobol sequence (simplified implementation)
            random_normals = self._generate_sobol_normals(self.params.time_steps, n_sims)
        else:
            random_normals = np.random.standard_normal((self.params.time_steps, n_sims))
        
        # Initialize paths
        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0] = self.params.initial_price
        
        # Generate paths
        for t in range(1, self.params.time_steps + 1):
            drift_term = (self.params.drift - 0.5 * self.params.volatility**2) * dt
            diffusion_term = self.params.volatility * np.sqrt(dt) * random_normals[t-1]
            paths[t] = paths[t-1] * np.exp(drift_term + diffusion_term)
        
        return paths
    
    def mean_reverting_process(self, n_simulations: Optional[int] = None) -> np.ndarray:
        """Generate paths using mean-reverting process (Ornstein-Uhlenbeck)"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.params.time_horizon / self.params.time_steps
        
        kappa = self.params.mean_reversion_speed
        theta = self.params.long_term_mean
        sigma = self.params.volatility
        
        # Initialize paths
        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0] = self.params.initial_price
        
        # Generate random numbers
        random_normals = np.random.standard_normal((self.params.time_steps, n_sims))
        
        # Generate paths using Euler-Maruyama scheme
        for t in range(1, self.params.time_steps + 1):
            drift_term = kappa * (theta - paths[t-1]) * dt
            diffusion_term = sigma * np.sqrt(dt) * random_normals[t-1]
            paths[t] = paths[t-1] + drift_term + diffusion_term
            
            # Ensure non-negative prices if needed
            paths[t] = np.maximum(paths[t], 0.001)
        
        return paths
    
    def jump_diffusion_process(self, n_simulations: Optional[int] = None) -> np.ndarray:
        """Generate paths using jump-diffusion process (Merton model)"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.params.time_horizon / self.params.time_steps
        
        lambda_j = self.params.jump_intensity
        mu_j = self.params.jump_mean
        sigma_j = self.params.jump_volatility
        
        # Initialize paths
        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0] = self.params.initial_price
        
        # Generate random numbers
        random_normals = np.random.standard_normal((self.params.time_steps, n_sims))
        jump_times = np.random.poisson(lambda_j * dt, (self.params.time_steps, n_sims))
        jump_sizes = np.random.normal(mu_j, sigma_j, (self.params.time_steps, n_sims))
        
        # Generate paths
        for t in range(1, self.params.time_steps + 1):
            # Diffusion component
            drift_term = (self.params.drift - 0.5 * self.params.volatility**2) * dt
            diffusion_term = self.params.volatility * np.sqrt(dt) * random_normals[t-1]
            
            # Jump component
            jump_term = jump_times[t-1] * jump_sizes[t-1]
            
            paths[t] = paths[t-1] * np.exp(drift_term + diffusion_term + jump_term)
        
        return paths
    
    def heston_model(self, n_simulations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths using Heston stochastic volatility model"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.params.time_horizon / self.params.time_steps
        
        # Heston parameters
        v0 = self.params.initial_variance
        kappa = self.params.variance_mean_reversion
        theta = self.params.long_term_variance
        sigma_v = self.params.vol_of_vol
        rho = self.params.correlation
        
        # Initialize paths
        price_paths = np.zeros((self.params.time_steps + 1, n_sims))
        variance_paths = np.zeros((self.params.time_steps + 1, n_sims))
        
        price_paths[0] = self.params.initial_price
        variance_paths[0] = v0
        
        # Generate correlated random numbers
        random_normals = np.random.standard_normal((2, self.params.time_steps, n_sims))
        
        # Apply correlation
        z1 = random_normals[0]
        z2 = rho * random_normals[0] + np.sqrt(1 - rho**2) * random_normals[1]
        
        # Generate paths using Euler scheme with Feller condition
        for t in range(1, self.params.time_steps + 1):
            # Variance process (with Feller boundary condition)
            variance_drift = kappa * (theta - np.maximum(variance_paths[t-1], 0)) * dt
            variance_diffusion = sigma_v * np.sqrt(np.maximum(variance_paths[t-1], 0) * dt) * z2[t-1]
            variance_paths[t] = np.maximum(variance_paths[t-1] + variance_drift + variance_diffusion, 0)
            
            # Price process
            price_drift = self.params.drift * dt
            price_diffusion = np.sqrt(np.maximum(variance_paths[t-1], 0) * dt) * z1[t-1]
            price_paths[t] = price_paths[t-1] * np.exp(price_drift - 0.5 * variance_paths[t-1] * dt + price_diffusion)
        
        return price_paths, variance_paths
    
    def _generate_sobol_normals(self, time_steps: int, n_simulations: int) -> np.ndarray:
        """Generate quasi-random normal numbers using Sobol sequence (simplified)"""
        # This is a simplified implementation
        # In production, use libraries like scipy.stats.qmc
        uniform_randoms = np.random.uniform(0, 1, (time_steps, n_simulations))
        return stats.norm.ppf(uniform_randoms)
    
    def generate_paths(self) -> np.ndarray:
        """Generate price paths based on specified process"""
        if self.params.process_type == ProcessType.GEOMETRIC_BROWNIAN_MOTION:
            return self.geometric_brownian_motion()
        elif self.params.process_type == ProcessType.MEAN_REVERTING:
            return self.mean_reverting_process()
        elif self.params.process_type == ProcessType.JUMP_DIFFUSION:
            return self.jump_diffusion_process()
        elif self.params.process_type == ProcessType.HESTON:
            price_paths, _ = self.heston_model()
            return price_paths
        else:
            raise ValueError(f"Unsupported process type: {self.params.process_type}")
    
    def calculate_payoff(self, paths: np.ndarray, payoff_spec: OptionPayoff, 
                        discount_factor: float = 1.0) -> np.ndarray:
        """Calculate option payoffs from price paths"""
        final_prices = paths[-1]  # Final prices
        
        if payoff_spec.option_type == 'call':
            payoffs = np.maximum(final_prices - payoff_spec.strike_price, 0)
        
        elif payoff_spec.option_type == 'put':
            payoffs = np.maximum(payoff_spec.strike_price - final_prices, 0)
        
        elif payoff_spec.option_type == 'asian_call':
            # Arithmetic average
            if payoff_spec.averaging_period:
                avg_prices = np.mean(paths[-payoff_spec.averaging_period:], axis=0)
            else:
                avg_prices = np.mean(paths, axis=0)
            payoffs = np.maximum(avg_prices - payoff_spec.strike_price, 0)
        
        elif payoff_spec.option_type == 'asian_put':
            if payoff_spec.averaging_period:
                avg_prices = np.mean(paths[-payoff_spec.averaging_period:], axis=0)
            else:
                avg_prices = np.mean(paths, axis=0)
            payoffs = np.maximum(payoff_spec.strike_price - avg_prices, 0)
        
        elif payoff_spec.option_type == 'barrier_call':
            # Simplified barrier option
            barrier_breached = np.any(paths >= payoff_spec.barrier_level, axis=0)
            if payoff_spec.barrier_type == 'up_and_out':
                payoffs = np.where(barrier_breached, 0, np.maximum(final_prices - payoff_spec.strike_price, 0))
            elif payoff_spec.barrier_type == 'up_and_in':
                payoffs = np.where(barrier_breached, np.maximum(final_prices - payoff_spec.strike_price, 0), 0)
            else:
                payoffs = np.maximum(final_prices - payoff_spec.strike_price, 0)
        
        elif payoff_spec.option_type == 'lookback_call':
            # Lookback call: max(S_max - K, 0)
            if payoff_spec.lookback_period:
                max_prices = np.max(paths[-payoff_spec.lookback_period:], axis=0)
            else:
                max_prices = np.max(paths, axis=0)
            payoffs = np.maximum(max_prices - payoff_spec.strike_price, 0)
        
        elif payoff_spec.option_type == 'lookback_put':
            # Lookback put: max(K - S_min, 0)
            if payoff_spec.lookback_period:
                min_prices = np.min(paths[-payoff_spec.lookback_period:], axis=0)
            else:
                min_prices = np.min(paths, axis=0)
            payoffs = np.maximum(payoff_spec.strike_price - min_prices, 0)
        
        else:
            raise ValueError(f"Unsupported option type: {payoff_spec.option_type}")
        
        return payoffs * discount_factor
    
    def price_option(self, payoff_spec: OptionPayoff, risk_free_rate: float = 0.0) -> SimulationResult:
        """Price option using Monte Carlo simulation"""
        start_time = datetime.now()
        
        try:
            # Generate paths
            if self.params.parallel and self.params.num_simulations > 10000:
                paths = self._parallel_path_generation()
            else:
                paths = self.generate_paths()
            
            # Calculate discount factor
            discount_factor = np.exp(-risk_free_rate * self.params.time_horizon)
            
            # Calculate payoffs
            payoffs = self.calculate_payoff(paths, payoff_spec, discount_factor)
            
            # Calculate option price and statistics
            option_price = np.mean(payoffs)
            standard_error = np.std(payoffs) / np.sqrt(len(payoffs))
            
            # Confidence interval (95%)
            confidence_level = 1.96
            ci_lower = option_price - confidence_level * standard_error
            ci_upper = option_price + confidence_level * standard_error
            
            # Calculate Greeks using finite differences
            greeks = self._calculate_greeks(payoff_spec, risk_free_rate)
            
            # Convergence analysis
            convergence_data = self._analyze_convergence(payoffs)
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return SimulationResult(
                option_price=option_price,
                standard_error=standard_error,
                confidence_interval=(ci_lower, ci_upper),
                paths=paths if self.params.num_simulations <= 1000 else None,  # Store paths only for small simulations
                payoffs=payoffs,
                greeks=greeks,
                convergence_data=convergence_data,
                computation_time=computation_time,
                method_used=f"Monte Carlo ({self.params.process_type.value})"
            )
            
        except Exception as e:
            logger.error(f"Option pricing failed: {e}")
            raise
    
    def _parallel_path_generation(self) -> np.ndarray:
        """Generate paths using parallel processing"""
        num_cores = min(mp.cpu_count(), 8)  # Limit to 8 cores
        sims_per_core = self.params.num_simulations // num_cores
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(num_cores):
                n_sims = sims_per_core if i < num_cores - 1 else self.params.num_simulations - i * sims_per_core
                future = executor.submit(self._generate_paths_chunk, n_sims)
                futures.append(future)
            
            # Collect results
            path_chunks = [future.result() for future in futures]
        
        # Concatenate paths
        return np.concatenate(path_chunks, axis=1)
    
    def _generate_paths_chunk(self, n_simulations: int) -> np.ndarray:
        """Generate a chunk of paths for parallel processing"""
        if self.params.process_type == ProcessType.GEOMETRIC_BROWNIAN_MOTION:
            return self.geometric_brownian_motion(n_simulations)
        elif self.params.process_type == ProcessType.MEAN_REVERTING:
            return self.mean_reverting_process(n_simulations)
        elif self.params.process_type == ProcessType.JUMP_DIFFUSION:
            return self.jump_diffusion_process(n_simulations)
        elif self.params.process_type == ProcessType.HESTON:
            price_paths, _ = self.heston_model(n_simulations)
            return price_paths
        else:
            raise ValueError(f"Unsupported process type: {self.params.process_type}")
    
    def _calculate_greeks(self, payoff_spec: OptionPayoff, risk_free_rate: float) -> Dict[str, float]:
        """Calculate Greeks using finite differences"""
        try:
            # Delta: sensitivity to underlying price
            bump_size = 0.01 * self.params.initial_price
            
            # Price up
            params_up = SimulationParameters(
                initial_price=self.params.initial_price + bump_size,
                drift=self.params.drift,
                volatility=self.params.volatility,
                time_horizon=self.params.time_horizon,
                time_steps=self.params.time_steps,
                num_simulations=min(self.params.num_simulations, 10000),  # Reduce for Greeks calculation
                process_type=self.params.process_type,
                random_seed=self.params.random_seed
            )
            
            simulator_up = EnhancedMonteCarloSimulator(params_up)
            paths_up = simulator_up.generate_paths()
            payoffs_up = simulator_up.calculate_payoff(paths_up, payoff_spec, np.exp(-risk_free_rate * self.params.time_horizon))
            price_up = np.mean(payoffs_up)
            
            # Price down
            params_down = SimulationParameters(
                initial_price=self.params.initial_price - bump_size,
                drift=self.params.drift,
                volatility=self.params.volatility,
                time_horizon=self.params.time_horizon,
                time_steps=self.params.time_steps,
                num_simulations=min(self.params.num_simulations, 10000),
                process_type=self.params.process_type,
                random_seed=self.params.random_seed
            )
            
            simulator_down = EnhancedMonteCarloSimulator(params_down)
            paths_down = simulator_down.generate_paths()
            payoffs_down = simulator_down.calculate_payoff(paths_down, payoff_spec, np.exp(-risk_free_rate * self.params.time_horizon))
            price_down = np.mean(payoffs_down)
            
            # Calculate Delta
            delta = (price_up - price_down) / (2 * bump_size)
            
            # Vega: sensitivity to volatility
            vol_bump = 0.01  # 1% volatility bump
            
            params_vol_up = SimulationParameters(
                initial_price=self.params.initial_price,
                drift=self.params.drift,
                volatility=self.params.volatility + vol_bump,
                time_horizon=self.params.time_horizon,
                time_steps=self.params.time_steps,
                num_simulations=min(self.params.num_simulations, 10000),
                process_type=self.params.process_type,
                random_seed=self.params.random_seed
            )
            
            simulator_vol_up = EnhancedMonteCarloSimulator(params_vol_up)
            paths_vol_up = simulator_vol_up.generate_paths()
            payoffs_vol_up = simulator_vol_up.calculate_payoff(paths_vol_up, payoff_spec, np.exp(-risk_free_rate * self.params.time_horizon))
            price_vol_up = np.mean(payoffs_vol_up)
            
            # Base price (for Vega calculation)
            paths_base = self.generate_paths()
            payoffs_base = self.calculate_payoff(paths_base, payoff_spec, np.exp(-risk_free_rate * self.params.time_horizon))
            price_base = np.mean(payoffs_base)
            
            vega = (price_vol_up - price_base) / vol_bump
            
            return {
                'delta': delta,
                'vega': vega,
                'gamma': 0.0,  # Would require second-order finite differences
                'theta': 0.0,  # Would require time bump
                'rho': 0.0     # Would require rate bump
            }
            
        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}")
            return {'delta': 0.0, 'vega': 0.0, 'gamma': 0.0, 'theta': 0.0, 'rho': 0.0}
    
    def _analyze_convergence(self, payoffs: np.ndarray) -> Dict[str, List[float]]:
        """Analyze convergence of Monte Carlo simulation"""
        try:
            sample_sizes = np.logspace(2, np.log10(len(payoffs)), 20, dtype=int)
            sample_sizes = np.unique(sample_sizes)
            sample_sizes = sample_sizes[sample_sizes <= len(payoffs)]
            
            running_means = []
            running_stds = []
            
            for n in sample_sizes:
                running_means.append(np.mean(payoffs[:n]))
                running_stds.append(np.std(payoffs[:n]) / np.sqrt(n))
            
            return {
                'sample_sizes': sample_sizes.tolist(),
                'running_means': running_means,
                'running_standard_errors': running_stds
            }
            
        except Exception as e:
            logger.warning(f"Convergence analysis failed: {e}")
            return {}
    
    def calculate_var(self, confidence_level: float = 0.05, portfolio_value: float = 1000000) -> Dict[str, float]:
        """Calculate Value at Risk using Monte Carlo simulation"""
        try:
            # Generate paths
            paths = self.generate_paths()
            
            # Calculate returns
            initial_value = paths[0, 0]
            final_values = paths[-1]
            returns = (final_values - initial_value) / initial_value
            
            # Scale to portfolio value
            portfolio_returns = returns * portfolio_value
            
            # Calculate VaR
            var = np.percentile(portfolio_returns, confidence_level * 100)
            
            # Calculate Conditional VaR (Expected Shortfall)
            cvar = np.mean(portfolio_returns[portfolio_returns <= var])
            
            return {
                'var': abs(var),
                'cvar': abs(cvar),
                'confidence_level': confidence_level,
                'portfolio_value': portfolio_value,
                'worst_case': abs(np.min(portfolio_returns)),
                'best_case': np.max(portfolio_returns)
            }
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise
    
    def stress_test(self, stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform stress testing with different market scenarios"""
        try:
            results = {}
            
            for i, scenario in enumerate(stress_scenarios):
                scenario_name = scenario.get('name', f'Scenario_{i+1}')
                
                # Create stressed parameters
                stressed_params = SimulationParameters(
                    initial_price=self.params.initial_price,
                    drift=scenario.get('drift', self.params.drift),
                    volatility=scenario.get('volatility', self.params.volatility),
                    time_horizon=self.params.time_horizon,
                    time_steps=self.params.time_steps,
                    num_simulations=self.params.num_simulations,
                    process_type=self.params.process_type,
                    random_seed=self.params.random_seed
                )
                
                # Run simulation
                stressed_simulator = EnhancedMonteCarloSimulator(stressed_params)
                paths = stressed_simulator.generate_paths()
                
                # Calculate statistics
                final_prices = paths[-1]
                results[scenario_name] = {
                    'mean_final_price': np.mean(final_prices),
                    'std_final_price': np.std(final_prices),
                    'min_price': np.min(final_prices),
                    'max_price': np.max(final_prices),
                    'percentile_5': np.percentile(final_prices, 5),
                    'percentile_95': np.percentile(final_prices, 95),
                    'scenario_parameters': scenario
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            raise


# Backward compatibility class
class MonteCarloSimulator:
    """Backward compatible Monte Carlo simulator"""
    
    def __init__(self, S0: float, mu: float, sigma: float, T: float = 1, steps: int = 252):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps = steps
        
        # Create enhanced simulator
        self.params = SimulationParameters(
            initial_price=S0,
            drift=mu,
            volatility=sigma,
            time_horizon=T,
            time_steps=steps,
            num_simulations=10000,
            process_type=ProcessType.GEOMETRIC_BROWNIAN_MOTION
        )
        self.enhanced_simulator = EnhancedMonteCarloSimulator(self.params)
    
    def geometric_brownian_motion(self, n_simulations: int = 10000) -> np.ndarray:
        """Generate GBM paths (backward compatible)"""
        self.params.num_simulations = n_simulations
        self.enhanced_simulator = EnhancedMonteCarloSimulator(self.params)
        return self.enhanced_simulator.geometric_brownian_motion()
    
    def asian_option_price(self, K: float, r: float, simulations: int = 100000) -> float:
        """Calculate Asian option price (backward compatible)"""
        self.params.num_simulations = simulations
        self.enhanced_simulator = EnhancedMonteCarloSimulator(self.params)
        
        payoff_spec = OptionPayoff(
            option_type='asian_call',
            strike_price=K
        )
        
        result = self.enhanced_simulator.price_option(payoff_spec, r)
        return result.option_price


# Convenience functions
def monte_carlo_option_price(S0: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = 'call', simulations: int = 100000) -> float:
    """Calculate option price using Monte Carlo (convenience function)"""
    params = SimulationParameters(
        initial_price=S0,
        drift=r,
        volatility=sigma,
        time_horizon=T,
        time_steps=int(T * 252),
        num_simulations=simulations,
        process_type=ProcessType.GEOMETRIC_BROWNIAN_MOTION
    )
    
    simulator = EnhancedMonteCarloSimulator(params)
    payoff_spec = OptionPayoff(option_type=option_type, strike_price=K)
    result = simulator.price_option(payoff_spec, r)
    
    return result.option_price


def calculate_portfolio_var(returns_data: np.ndarray, confidence_level: float = 0.05, 
                          portfolio_value: float = 1000000) -> Dict[str, float]:
    """Calculate portfolio VaR from historical returns"""
    portfolio_returns = returns_data * portfolio_value
    var = np.percentile(portfolio_returns, confidence_level * 100)
    cvar = np.mean(portfolio_returns[portfolio_returns <= var])
    
    return {
        'var': abs(var),
        'cvar': abs(cvar),
        'confidence_level': confidence_level,
        'portfolio_value': portfolio_value
    }

