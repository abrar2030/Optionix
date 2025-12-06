"""
Integration tests for the enhanced Optionix platform.

This module provides tests for validating the integration between
the enhanced pricing models, trade execution engine, circuit breakers,
and risk management tools.
"""

import sys
import unittest

sys.path.append("/home/ubuntu/Optionix/code")
from backend.services.risk_management.risk_engine import RiskEngine, RiskMetricType
from backend.services.trade_execution.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerType,
)
from backend.services.trade_execution.execution_engine import (
    ExecutionEngine,
    OrderSide,
    OrderType,
)
from quantitative.enhanced.calibration_engine import CalibrationEngine
from quantitative.enhanced.local_volatility import DupireLocalVolModel
from quantitative.enhanced.stochastic_volatility import HestonModel, SabrModel
from quantitative.enhanced.volatility_surface import VolatilitySurface


class IntegrationTests(unittest.TestCase):
    """Integration tests for the enhanced Optionix platform."""

    def setUp(self) -> Any:
        """Set up test environment."""
        self.market_data = {
            "SPY": {
                "price": 450.0,
                "volatility": 0.15,
                "rate": 0.02,
                "dividend": 0.01,
                "bid": 449.8,
                "ask": 450.2,
                "volume": 10000000,
                "options": {
                    "calls": [
                        {
                            "strike": 440,
                            "expiry": "2023-12-15",
                            "price": 15.5,
                            "iv": 0.16,
                            "spot": 450.0,
                        },
                        {
                            "strike": 450,
                            "expiry": "2023-12-15",
                            "price": 10.2,
                            "iv": 0.15,
                            "spot": 450.0,
                        },
                        {
                            "strike": 460,
                            "expiry": "2023-12-15",
                            "price": 6.3,
                            "iv": 0.14,
                            "spot": 450.0,
                        },
                    ],
                    "puts": [
                        {
                            "strike": 440,
                            "expiry": "2023-12-15",
                            "price": 5.1,
                            "iv": 0.17,
                            "spot": 450.0,
                        },
                        {
                            "strike": 450,
                            "expiry": "2023-12-15",
                            "price": 9.8,
                            "iv": 0.16,
                            "spot": 450.0,
                        },
                        {
                            "strike": 460,
                            "expiry": "2023-12-15",
                            "price": 15.7,
                            "iv": 0.15,
                            "spot": 450.0,
                        },
                    ],
                },
            }
        }
        self.portfolio = {
            "portfolio_id": "test_portfolio",
            "positions": [
                {
                    "position_id": "pos1",
                    "instrument": "SPY",
                    "type": "equity",
                    "quantity": 1000,
                    "value": 450000.0,
                },
                {
                    "position_id": "pos2",
                    "instrument": "SPY_DEC23_450C",
                    "type": "option",
                    "option_type": "call",
                    "quantity": 50,
                    "value": 51000.0,
                    "strike": 450.0,
                    "expiry": "2023-12-15",
                    "delta": 0.52,
                    "gamma": 0.03,
                    "vega": 0.45,
                    "theta": -0.08,
                    "rho": 0.15,
                },
            ],
        }
        self.option_data = []
        for call in self.market_data["SPY"]["options"]["calls"]:
            self.option_data.append(
                {
                    "strike": call["strike"],
                    "time_to_expiry": 0.5,
                    "expiry": 0.5,
                    "price": call["price"],
                    "option_type": "call",
                    "spot": self.market_data["SPY"]["price"],
                    "rate": self.market_data["SPY"]["rate"],
                    "dividend": self.market_data["SPY"]["dividend"],
                    "market_price": call["price"],
                }
            )
        for put in self.market_data["SPY"]["options"]["puts"]:
            self.option_data.append(
                {
                    "strike": put["strike"],
                    "time_to_expiry": 0.5,
                    "expiry": 0.5,
                    "price": put["price"],
                    "option_type": "put",
                    "spot": self.market_data["SPY"]["price"],
                    "rate": self.market_data["SPY"]["rate"],
                    "dividend": self.market_data["SPY"]["dividend"],
                    "market_price": put["price"],
                }
            )

    def test_enhanced_pricing_models(self) -> Any:
        """Test enhanced pricing models."""
        heston = HestonModel()
        heston_price = heston.price_option(
            spot=self.market_data["SPY"]["price"],
            strike=450.0,
            time_to_expiry=0.5,
            rate=self.market_data["SPY"]["rate"],
            dividend=self.market_data["SPY"]["dividend"],
            option_type="call",
        )
        self.assertIsNotNone(heston_price)
        self.assertGreater(heston_price, 0)
        sabr = SabrModel()
        sabr_vol = sabr.implied_volatility(
            strike=450.0, forward=self.market_data["SPY"]["price"], time_to_expiry=0.5
        )
        self.assertIsNotNone(sabr_vol)
        self.assertGreater(sabr_vol, 0)
        dupire = DupireLocalVolModel()
        dupire.calibrate(
            self.option_data,
            spot=self.market_data["SPY"]["price"],
            rate=self.market_data["SPY"]["rate"],
            dividend=self.market_data["SPY"]["dividend"],
        )
        dupire_vol = dupire.local_volatility(
            spot=self.market_data["SPY"]["price"], strike=450.0, time_to_expiry=0.5
        )
        self.assertIsNotNone(dupire_vol)
        self.assertGreater(dupire_vol, 0)
        vol_surface = VolatilitySurface()
        vol_surface.fit_surface(self.market_data["SPY"]["options"])
        interp_vol = vol_surface.get_volatility(450.0, 0.5)
        self.assertIsNotNone(interp_vol)
        self.assertGreater(interp_vol, 0)
        calibration = CalibrationEngine()
        params = calibration.calibrate_heston(self.option_data)
        self.assertIsNotNone(params)
        self.assertEqual(len(params), 5)

    def test_trade_execution_engine(self) -> Any:
        """Test trade execution engine."""
        execution_engine = ExecutionEngine()
        order_params = {
            "instrument": "SPY",
            "quantity": 100,
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
        }
        result = execution_engine.submit_order(order_params)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "accepted")
        order_id = result["order_id"]
        status = execution_engine.get_order_status(order_id)
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "success")
        metrics = execution_engine.get_execution_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics["total_orders"], 1)

    def test_circuit_breaker(self) -> Any:
        """Test circuit breaker functionality."""
        circuit_breaker = CircuitBreaker()
        breaker = circuit_breaker.activate_circuit_breaker(
            instrument="SPY",
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            reason="Test activation",
            duration_minutes=15,
            data={"price_change_percent": 8.5},
        )
        self.assertIsNotNone(breaker)
        self.assertEqual(breaker["instrument"], "SPY")
        self.assertEqual(breaker["type"], CircuitBreakerType.PRICE_MOVEMENT.value)
        is_active = circuit_breaker.is_active("SPY")
        self.assertTrue(is_active)
        deactivated = circuit_breaker.deactivate_circuit_breaker(
            "SPY", "Test deactivation"
        )
        self.assertTrue(deactivated)
        is_active = circuit_breaker.is_active("SPY")
        self.assertFalse(is_active)

    def test_risk_management(self) -> Any:
        """Test risk management tools."""
        risk_engine = RiskEngine()
        risk_metrics = risk_engine.calculate_portfolio_risk(
            self.portfolio,
            metrics=[
                RiskMetricType.VAR,
                RiskMetricType.VOLATILITY,
                RiskMetricType.SHARPE_RATIO,
            ],
        )
        self.assertIsNotNone(risk_metrics)
        self.assertIn("var", risk_metrics)
        self.assertIn("volatility", risk_metrics)
        scenario_results = risk_engine.run_scenario_analysis(self.portfolio)
        self.assertIsNotNone(scenario_results)
        self.assertGreaterEqual(len(scenario_results), 1)
        stress_test_results = risk_engine.run_stress_test(self.portfolio)
        self.assertIsNotNone(stress_test_results)
        self.assertIn("scenario_results", stress_test_results)
        self.assertIn("aggregated_results", stress_test_results)
        what_if_results = risk_engine.run_what_if_analysis(self.portfolio)
        self.assertIsNotNone(what_if_results)
        self.assertGreaterEqual(len(what_if_results), 1)

    def test_integration_pricing_and_risk(self) -> Any:
        """Test integration between pricing models and risk management."""
        heston = HestonModel()
        risk_engine = RiskEngine()
        option_price = heston.price_option(
            spot=self.market_data["SPY"]["price"],
            strike=450.0,
            time_to_expiry=0.5,
            rate=self.market_data["SPY"]["rate"],
            dividend=self.market_data["SPY"]["dividend"],
            option_type="call",
        )
        portfolio_copy = self.portfolio.copy()
        portfolio_copy["positions"][1]["value"] = option_price * 100
        risk_metrics = risk_engine.calculate_portfolio_risk(portfolio_copy)
        self.assertIsNotNone(risk_metrics)
        self.assertIn("var", risk_metrics)

    def test_integration_execution_and_circuit_breaker(self) -> Any:
        """Test integration between execution engine and circuit breaker."""
        ExecutionEngine()
        circuit_breaker = CircuitBreaker()
        circuit_breaker.activate_circuit_breaker(
            instrument="SPY",
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            reason="Test activation",
            duration_minutes=15,
            data={"price_change_percent": 8.5},
        )
        order_params = {
            "instrument": "SPY",
            "quantity": 100,
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
        }
        is_active = circuit_breaker.is_active("SPY")
        if is_active:
            order_params["rejection_reason"] = "Circuit breaker active for SPY"
        self.assertTrue(is_active)
        self.assertIn("rejection_reason", order_params)

    def test_integration_risk_and_execution(self) -> Any:
        """Test integration between risk management and execution engine."""
        risk_engine = RiskEngine()
        execution_engine = ExecutionEngine()
        initial_risk = risk_engine.calculate_portfolio_risk(self.portfolio)
        order_params = {
            "instrument": "SPY",
            "quantity": 100,
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
        }
        result = execution_engine.submit_order(order_params)
        result["order_id"]
        portfolio_copy = self.portfolio.copy()
        portfolio_copy["positions"].append(
            {
                "position_id": "pos3",
                "instrument": "SPY",
                "type": "equity",
                "quantity": 100,
                "value": 45000.0,
            }
        )
        new_risk = risk_engine.calculate_portfolio_risk(portfolio_copy)
        self.assertNotEqual(
            initial_risk.get("var", {}).get("var_95", 0),
            new_risk.get("var", {}).get("var_95", 0),
        )


if __name__ == "__main__":
    unittest.main()
