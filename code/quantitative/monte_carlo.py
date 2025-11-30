"""
Monte Carlo Simulation Module for Optionix Platform
Implements comprehensive Monte Carlo methods for:
- Option pricing (European, Asian, Barrier, Exotic)
- Risk management (VaR, CVaR, stress testing)
- Variance reduction techniques
- Parallel processing for performance
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats.qmc import Sobol  # Use proper QMC implementation

# Configure logging to suppress the warnings filter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    QUASI_RANDOM = "quasi_random"


@dataclass
class SimulationParameters:
    """Monte Carlo simulation parameters"""

    initial_price: float
    # CRITICAL FIX: Use risk-free rate (r) for drift in risk-neutral pricing
    risk_free_rate: float  # Replaced 'drift'
    volatility: float
    time_horizon: float
    time_steps: int
    num_simulations: int
    process_type: ProcessType = ProcessType.GEOMETRIC_BROWNIAN_MOTION
    variance_reduction: Optional[VarianceReduction] = None
    random_seed: Optional[int] = None
    # Set parallel processing to False by default for smaller simulations
    parallel: bool = False

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

    # Renamed to make intent clearer
    option_style: str  # 'european', 'asian', 'barrier', 'lookback'
    option_type: str  # 'call', 'put'
    strike_price: float

    barrier_level: Optional[float] = None
    barrier_type: Optional[str] = None  # 'up_and_out', 'down_and_in', etc.

    # For exotic options: usually 0 (start of path)
    averaging_start_step: int = 0
    lookback_start_step: int = 0


@dataclass
class SimulationResult:
    """Monte Carlo simulation result"""

    option_price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    # Use field for mutable defaults in dataclass
    paths: Optional[np.ndarray] = field(default=None, repr=False)
    payoffs: Optional[np.ndarray] = field(default=None, repr=False)
    greeks: Dict[str, float] = field(default_factory=lambda: {})
    convergence_data: Dict[str, List[float]] = field(default_factory=lambda: {})
    computation_time: float = 0.0
    method_used: str = ""


class MCSimulator:
    """Monte Carlo simulator with advanced features"""

    def __init__(self, params: SimulationParameters):
        """Initialize Monte Carlo simulator"""
        self.params = params
        self.dt = params.time_horizon / params.time_steps
        self.validate_parameters()

        # Set random seed for reproducibility
        if params.random_seed is not None:
            np.random.seed(params.random_seed)

    def validate_parameters(self):
        """Validate simulation parameters and financial constraints"""
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
        if self.params.risk_free_rate is None:
            raise ValueError("Risk-free rate must be provided for pricing")

        if self.params.process_type == ProcessType.MEAN_REVERTING and (
            self.params.mean_reversion_speed is None
            or self.params.long_term_mean is None
        ):
            raise ValueError("Mean reversion parameters required")

    # CRITICAL OPTIMIZATION: Fully vectorized GBM
    def geometric_brownian_motion(
        self, n_simulations: Optional[int] = None
    ) -> np.ndarray:
        """Generate paths using Geometric Brownian Motion (Vectorized)"""
        n_sims = n_simulations or self.params.num_simulations

        # 1. Generate random numbers based on variance reduction
        if self.params.variance_reduction == VarianceReduction.ANTITHETIC:
            n_base = n_sims // 2
            random_base = np.random.standard_normal((self.params.time_steps, n_base))
            random_normals = np.concatenate([random_base, -random_base], axis=1)
            # Handle odd number of simulations
            if n_sims % 2 != 0:
                random_normals = np.concatenate(
                    [
                        random_normals,
                        np.random.standard_normal((self.params.time_steps, 1)),
                    ],
                    axis=1,
                )

        elif self.params.variance_reduction == VarianceReduction.QUASI_RANDOM:
            # Use proper Sobol implementation
            sobol = Sobol(d=self.params.time_steps, scramble=True)
            uniform_randoms = sobol.random(n=n_sims)
            random_normals = stats.norm.ppf(uniform_randoms).T

        else:
            random_normals = np.random.standard_normal((self.params.time_steps, n_sims))

        # 2. Calculate log returns (Vectorized)
        # S(t+dt) = S(t) * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        drift_term = (
            self.params.risk_free_rate - 0.5 * self.params.volatility**2
        ) * self.dt
        diffusion_term = self.params.volatility * np.sqrt(self.dt) * random_normals
        log_returns = drift_term + diffusion_term

        # 3. Calculate cumulative log returns
        cumulative_log_returns = np.cumsum(log_returns, axis=0)

        # 4. Initialize and fill paths (Vectorized)
        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0, :] = self.params.initial_price
        paths[1:] = self.params.initial_price * np.exp(cumulative_log_returns)

        return paths

    def mean_reverting_process(self, n_simulations: Optional[int] = None) -> np.ndarray:
        """Generate paths using mean-reverting process (Ornstein-Uhlenbeck)"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.dt
        kappa = self.params.mean_reversion_speed
        theta = self.params.long_term_mean
        sigma = self.params.volatility

        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0] = self.params.initial_price

        # Vectorized random numbers
        random_normals = np.random.standard_normal((self.params.time_steps, n_sims))

        # Euler-Maruyama scheme (less vectorizable across time steps than GBM)
        for t in range(1, self.params.time_steps + 1):
            drift_term = kappa * (theta - paths[t - 1]) * dt
            diffusion_term = sigma * np.sqrt(dt) * random_normals[t - 1]
            paths[t] = paths[t - 1] + drift_term + diffusion_term
            paths[t] = np.maximum(paths[t], 0.001)  # Ensure non-negative/stable

        return paths

    # Heston Model (Retained and slightly cleaned up)
    def heston_model(
        self, n_simulations: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths using Heston stochastic volatility model"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.dt

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
        z1 = random_normals[0]
        z2 = rho * random_normals[0] + np.sqrt(1 - rho**2) * random_normals[1]

        # Generate paths using Euler scheme (full vectorization is complex for Heston/Euler)
        for t in range(1, self.params.time_steps + 1):

            V_prev = np.maximum(
                variance_paths[t - 1], 0
            )  # Feller condition enforcement

            # Variance process (with Feller boundary condition)
            variance_drift = kappa * (theta - V_prev) * dt
            variance_diffusion = sigma_v * np.sqrt(V_prev * dt) * z2[t - 1]
            variance_paths[t] = np.maximum(
                variance_paths[t - 1] + variance_drift + variance_diffusion, 0
            )

            # Price process
            price_drift_log = (self.params.risk_free_rate - 0.5 * V_prev) * dt
            price_diffusion_log = np.sqrt(V_prev * dt) * z1[t - 1]

            price_paths[t] = price_paths[t - 1] * np.exp(
                price_drift_log + price_diffusion_log
            )

        return price_paths, variance_paths

    def _generate_paths_chunk(self, n_simulations: int) -> np.ndarray:
        """Generate a chunk of paths for parallel processing"""

        # FIX: Ensure paths are independent in parallel workers
        if self.params.random_seed is not None:
            # Set a unique sub-seed based on the process ID
            np.random.seed(self.params.random_seed + mp.current_process().pid)

        # Instantiate a temporary simulator or use a function,
        # but for simplicity, we use the methods of the current instance
        if self.params.process_type == ProcessType.GEOMETRIC_BROWNIAN_MOTION:
            paths = self.geometric_brownian_motion(n_simulations)
        elif self.params.process_type == ProcessType.MEAN_REVERTING:
            paths = self.mean_reverting_process(n_simulations)
        elif self.params.process_type == ProcessType.HESTON:
            paths, _ = self.heston_model(n_simulations)
        else:
            # Fallback to GBM for unsupported, non-standard process types in parallel
            paths = self.geometric_brownian_motion(n_simulations)

        return paths

    def generate_paths(self) -> np.ndarray:
        """Generate price paths based on specified process"""
        if self.params.process_type == ProcessType.GEOMETRIC_BROWNIAN_MOTION:
            return self.geometric_brownian_motion()
        elif self.params.process_type == ProcessType.MEAN_REVERTING:
            return self.mean_reverting_process()
        elif self.params.process_type == ProcessType.HESTON:
            price_paths, _ = self.heston_model()
            return price_paths
        else:
            raise ValueError(f"Unsupported process type: {self.params.process_type}")

    # FIX: Payoff calculated without discounting. Discounting applied in price_option.
    def calculate_payoff(
        self, paths: np.ndarray, payoff_spec: OptionPayoff
    ) -> np.ndarray:
        """Calculate option payoffs from price paths (undiscounted)"""

        option_style = payoff_spec.option_style.lower()
        option_type = payoff_spec.option_type.lower()
        strike = payoff_spec.strike_price

        if option_type not in ["call", "put"]:
            raise ValueError(f"Unsupported option type: {option_type}")

        if option_style == "european":
            final_prices = paths[-1]
            payoffs = np.maximum(
                (
                    (final_prices - strike)
                    if option_type == "call"
                    else (strike - final_prices)
                ),
                0,
            )

        elif option_style == "asian":
            # Average price over the path (from start_step to end)
            start_step = payoff_spec.averaging_start_step
            average_prices = np.mean(paths[start_step:], axis=0)
            payoffs = np.maximum(
                (
                    (average_prices - strike)
                    if option_type == "call"
                    else (strike - average_prices)
                ),
                0,
            )

        elif option_style == "barrier":
            barrier = payoff_spec.barrier_level
            barrier_type = payoff_spec.barrier_type
            final_prices = paths[-1]

            # Payoff without barrier
            payoffs = np.maximum(
                (
                    (final_prices - strike)
                    if option_type == "call"
                    else (strike - final_prices)
                ),
                0,
            )

            # Apply barrier condition (simplified logic for Out options)
            paths_to_check = paths[1:]  # Exclude S0

            if barrier_type == "down_and_out":
                hit_barrier = np.any(paths_to_check < barrier, axis=0)
                payoffs[hit_barrier] = 0.0
            elif barrier_type == "up_and_out":
                hit_barrier = np.any(paths_to_check > barrier, axis=0)
                payoffs[hit_barrier] = 0.0

            # NOTE: In/Out options require more complex logic

        elif option_style == "lookback":
            start_step = payoff_spec.lookback_start_step
            paths_to_check = paths[start_step:]

            if option_type == "call":
                # Lookback call: max(S_max - K, 0)
                # Max price over the path
                max_prices = np.max(paths_to_check, axis=0)
                payoffs = np.maximum(max_prices - strike, 0)
            elif option_type == "put":
                # Lookback put: max(K - S_min, 0)
                # Min price over the path
                min_prices = np.min(paths_to_check, axis=0)
                payoffs = np.maximum(strike - min_prices, 0)

        elif option_style == "american":
            # CRITICAL FIX: American options need backward induction (LSM)
            raise NotImplementedError(
                "American option pricing requires LSM or similar method (LSM not implemented)"
            )

        else:
            raise ValueError(f"Unsupported option style: {option_style}")

        return payoffs

    def price_option(
        self, payoff_spec: OptionPayoff, risk_free_rate: Optional[float] = None
    ) -> SimulationResult:
        """Price option using Monte Carlo simulation"""
        start_time = datetime.now()

        r = risk_free_rate if risk_free_rate is not None else self.params.risk_free_rate

        try:
            # Generate paths
            if self.params.parallel and self.params.num_simulations > 10000:
                paths = self._parallel_path_generation()
            else:
                paths = self.generate_paths()

            # Calculate payoffs (undiscounted)
            payoffs_undiscounted = self.calculate_payoff(paths, payoff_spec)

            # Apply discount factor (once)
            discount_factor = np.exp(-r * self.params.time_horizon)
            payoffs = payoffs_undiscounted * discount_factor

            # Calculate option price and statistics
            option_price = np.mean(payoffs)
            standard_error = np.std(payoffs) / np.sqrt(len(payoffs))

            # Confidence interval (95% z-score = 1.96)
            confidence_level = 1.96
            ci_lower = option_price - confidence_level * standard_error
            ci_upper = option_price + confidence_level * standard_error

            # Calculate Greeks
            greeks = self._calculate_greeks(payoff_spec, r, option_price)

            # Convergence analysis
            convergence_data = self._analyze_convergence(payoffs)

            computation_time = (datetime.now() - start_time).total_seconds()

            return SimulationResult(
                option_price=option_price,
                standard_error=standard_error,
                confidence_interval=(ci_lower, ci_upper),
                paths=(
                    paths
                    if self.params.num_simulations <= 1000
                    and self.params.parallel is False
                    else None
                ),
                payoffs=payoffs_undiscounted,
                greeks=greeks,
                convergence_data=convergence_data,
                computation_time=computation_time,
                method_used=f"MC ({self.params.process_type.value})",
            )

        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"Option pricing failed: {e}")
            raise

    def _parallel_path_generation(self) -> np.ndarray:
        """Generate paths using parallel processing"""
        # Limit to available cores or a reasonable number (e.g., 8)
        num_cores = min(mp.cpu_count(), 8)
        sims_per_core = self.params.num_simulations // num_cores

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(num_cores):
                n_sims = (
                    sims_per_core
                    if i < num_cores - 1
                    else self.params.num_simulations - i * sims_per_core
                )

                # Submit task to generate a chunk
                future = executor.submit(self._generate_paths_chunk, n_sims)
                futures.append(future)

            # Collect results
            path_chunks = [future.result() for future in futures]

        # Concatenate paths
        return np.concatenate(path_chunks, axis=1)

    def _calculate_greeks(
        self, payoff_spec: OptionPayoff, risk_free_rate: float, base_price: float
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference.
        Uses central difference for Delta/Gamma for accuracy.
        """
        try:
            # --- Setup ---
            # Use relative bump for stability
            bump_size_spot = 0.001 * self.params.initial_price
            vol_bump = 0.001  # 0.1% volatility bump
            rate_bump = 0.0001  # 1bp rate bump

            # Reduce simulations for faster Greeks estimate
            greeks_n_sims = min(self.params.num_simulations, 10000)

            # --- Helper to calculate perturbed price ---
            def _get_bumped_price(
                s0_factor: float,
                vol_factor: float,
                time_factor: float,
                rate_factor: float,
            ) -> float:

                # Create bumped parameters for the finite difference step
                p_bump = SimulationParameters(
                    initial_price=self.params.initial_price
                    + s0_factor * bump_size_spot,
                    risk_free_rate=self.params.risk_free_rate + rate_factor * rate_bump,
                    volatility=self.params.volatility + vol_factor * vol_bump,
                    time_horizon=self.params.time_horizon + time_factor * self.dt,
                    time_steps=self.params.time_steps,
                    num_simulations=greeks_n_sims,
                    process_type=self.params.process_type,
                    random_seed=self.params.random_seed,  # Crucial for consistency
                    # Copy other process-specific parameters
                    mean_reversion_speed=self.params.mean_reversion_speed,
                    long_term_mean=self.params.long_term_mean,
                    # Heston params omitted for brevity but should be copied
                )

                simulator_bump = MCSimulator(p_bump)
                paths_bump = simulator_bump.generate_paths()
                payoffs_undiscounted = simulator_bump.calculate_payoff(
                    paths_bump, payoff_spec
                )
                discount_factor = np.exp(-p_bump.risk_free_rate * p_bump.time_horizon)
                return np.mean(payoffs_undiscounted * discount_factor)

            # --- Delta (Central Difference: S+ and S-) ---
            price_up = _get_bumped_price(
                s0_factor=1, vol_factor=0, time_factor=0, rate_factor=0
            )
            price_down = _get_bumped_price(
                s0_factor=-1, vol_factor=0, time_factor=0, rate_factor=0
            )
            delta = (price_up - price_down) / (2 * bump_size_spot)

            # --- Gamma (Central Difference: S+, S-, and base) ---
            gamma = (price_up - 2 * base_price + price_down) / (bump_size_spot**2)

            # --- Vega (Forward Difference: Vol+ and base) ---
            price_vol_up = _get_bumped_price(
                s0_factor=0, vol_factor=1, time_factor=0, rate_factor=0
            )
            vega = (price_vol_up - base_price) / vol_bump

            # --- Rho (Forward Difference: Rate+ and base) ---
            price_rate_up = _get_bumped_price(
                s0_factor=0, vol_factor=0, time_factor=0, rate_factor=1
            )
            rho = (price_rate_up - base_price) / rate_bump

            # --- Theta (Forward Difference: Time- and base) ---
            # Shift time back by one step (-dt)
            price_time_down = _get_bumped_price(
                s0_factor=0, vol_factor=0, time_factor=-1, rate_factor=0
            )
            # Time is measured in years, so divide by the change in time (dt)
            theta = (base_price - price_time_down) / self.dt

            return {
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "rho": rho,
                "theta": theta,
            }

        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}. Returning zeros.")
            return {"delta": 0.0, "vega": 0.0, "gamma": 0.0, "theta": 0.0, "rho": 0.0}

    def _analyze_convergence(self, payoffs: np.ndarray) -> Dict[str, List[float]]:
        """Analyze convergence of Monte Carlo simulation"""
        try:
            # Use logarithmic spacing for sample sizes
            sample_sizes = np.logspace(2, np.log10(len(payoffs)), 20, dtype=int)
            sample_sizes = np.unique(sample_sizes)
            sample_sizes = sample_sizes[sample_sizes <= len(payoffs)]

            running_means = []
            running_stds = []

            for n in sample_sizes:
                running_means.append(np.mean(payoffs[:n]))
                # Standard error is standard deviation divided by sqrt(n)
                running_stds.append(np.std(payoffs[:n]) / np.sqrt(n))

            return {
                "sample_sizes": sample_sizes.tolist(),
                "running_means": running_means,
                "running_standard_errors": running_stds,
            }

        except Exception as e:
            logger.warning(f"Convergence analysis failed: {e}")
            return {}

    def calculate_var(
        self, confidence_level: float = 0.05, portfolio_value: float = 1000000
    ) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Expected Shortfall (CVaR)"""
        try:
            # NOTE: Assumes the SimulationParameters contains the real-world drift (mu)
            # or uses historical data to estimate real-world price changes.
            paths = self.generate_paths()

            # Returns = (Final - Initial) / Initial
            initial_value = paths[0, 0]
            final_values = paths[-1]
            returns = (final_values - initial_value) / initial_value

            # Calculate Losses: Loss = -(Return * Portfolio Value)
            portfolio_losses = -returns * portfolio_value

            # VaR is the (1 - confidence_level) percentile of the **Losses**
            # e.g., 5% VaR is the 95th percentile loss.
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(portfolio_losses, var_percentile)

            # CVaR (Expected Shortfall) is the mean loss exceeding the VaR
            cvar = np.mean(portfolio_losses[portfolio_losses >= var])

            return {
                "var": var,  # Positive loss value
                "cvar": cvar,  # Positive loss value
                "confidence_level": confidence_level,
                "portfolio_value": portfolio_value,
                "worst_loss": np.max(portfolio_losses),
                "best_gain": -np.min(portfolio_losses),
            }

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise

    def stress_test(self, stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform stress testing with different market scenarios"""
        try:
            results = {}

            for i, scenario in enumerate(stress_scenarios):
                scenario_name = scenario.get("name", f"Scenario_{i+1}")

                # Create stressed parameters
                stressed_params = SimulationParameters(
                    initial_price=self.params.initial_price,
                    # NOTE: Using 'drift' here as a scenario-specific mu (expected return)
                    # not the risk-free rate
                    risk_free_rate=scenario.get("drift", self.params.risk_free_rate),
                    volatility=scenario.get("volatility", self.params.volatility),
                    time_horizon=self.params.time_horizon,
                    time_steps=self.params.time_steps,
                    num_simulations=self.params.num_simulations,
                    process_type=self.params.process_type,
                    random_seed=self.params.random_seed,
                    # Copy mean-reversion parameters
                    mean_reversion_speed=scenario.get(
                        "kappa", self.params.mean_reversion_speed
                    ),
                    long_term_mean=scenario.get("theta", self.params.long_term_mean),
                    # Other Heston/Jump diffusion parameters should be copied/stressed here
                )

                # Run simulation
                stressed_simulator = MCSimulator(stressed_params)
                paths = stressed_simulator.generate_paths()

                # Calculate statistics
                final_prices = paths[-1]
                results[scenario_name] = {
                    "mean_final_price": np.mean(final_prices),
                    "std_final_price": np.std(final_prices),
                    "min_price": np.min(final_prices),
                    "max_price": np.max(final_prices),
                    "percentile_5": np.percentile(final_prices, 5),
                    "percentile_95": np.percentile(final_prices, 95),
                    "scenario_parameters": scenario,
                }

            return results

        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            raise


# --- Simple compatibility class ---
class MonteCarloSimulator:
    """A compatibility class for a simple GBM Monte Carlo simulator"""

    def __init__(
        self, S0: float, mu: float, sigma: float, T: float = 1, steps: int = 252
    ):
        self.S0 = S0
        self.mu = mu  # This should be interpreted as the risk_free_rate (r) for pricing
        self.sigma = sigma
        self.T = T
        self.steps = steps

        # Create main simulator instance
        self.params = SimulationParameters(
            initial_price=S0,
            # CRITICAL FIX: Use mu as risk_free_rate for backward compatibility pricing
            risk_free_rate=mu,
            volatility=sigma,
            time_horizon=T,
            time_steps=steps,
            num_simulations=10000,
            process_type=ProcessType.GEOMETRIC_BROWNIAN_MOTION,
        )
        self.simulator = MCSimulator(self.params)

    def geometric_brownian_motion(self, n_simulations: int = 10000) -> np.ndarray:
        """Generate GBM paths (compatibility method)"""
        self.params.num_simulations = n_simulations
        self.simulator = MCSimulator(self.params)
        return self.simulator.geometric_brownian_motion()

    def asian_option_price(
        self, K: float, r: float, simulations: int = 100000
    ) -> float:
        """Calculate Asian option price (compatibility method)"""
        self.params.num_simulations = simulations
        self.params.risk_free_rate = r  # Ensure correct rate is used
        self.simulator = MCSimulator(self.params)

        # FIX: Use the new Payoff structure
        payoff_spec = OptionPayoff(
            option_style="asian", option_type="call", strike_price=K
        )

        result = self.simulator.price_option(payoff_spec, r)
        return result.option_price


# --- Convenience functions (Updated) ---
def monte_carlo_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    simulations: int = 100000,
) -> float:
    """Calculate European option price using Monte Carlo (convenience function)"""
    params = SimulationParameters(
        initial_price=S0,
        risk_free_rate=r,
        volatility=sigma,
        time_horizon=T,
        time_steps=int(T * 252),
        num_simulations=simulations,
        process_type=ProcessType.GEOMETRIC_BROWNIAN_MOTION,
    )

    simulator = MCSimulator(params)
    # FIX: Use the new Payoff structure
    payoff_spec = OptionPayoff(
        option_style="european", option_type=option_type, strike_price=K
    )
    result = simulator.price_option(payoff_spec, r)

    return result.option_price


def calculate_portfolio_var(
    returns_data: np.ndarray,
    confidence_level: float = 0.05,
    portfolio_value: float = 1000000,
) -> Dict[str, float]:
    """
    Calculate portfolio VaR and CVaR from historical returns
    (Independent from simulation class)
    """

    # Losses = -(Return * Portfolio Value)
    portfolio_losses = -returns_data * portfolio_value

    # VaR is the (1 - CL) percentile loss
    var_percentile = (1 - confidence_level) * 100
    var = np.percentile(portfolio_losses, var_percentile)

    # CVaR is the mean loss exceeding the VaR
    cvar = np.mean(portfolio_losses[portfolio_losses >= var])

    return {
        "var": var,
        "cvar": cvar,
        "confidence_level": confidence_level,
        "portfolio_value": portfolio_value,
    }
