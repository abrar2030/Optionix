import numpy as np


class RiskCalculator:

    @staticmethod
    def calculate_var(returns: Any, confidence: Any = 0.95) -> Any:
        return np.percentile(returns, 100 * (1 - confidence))

    @staticmethod
    def expected_shortfall(returns: Any, confidence: Any = 0.95) -> Any:
        var = RiskCalculator.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def margin_requirement(portfolio_value: Any, volatility: Any) -> Any:
        return portfolio_value * (0.1 + 0.9 * (volatility / 0.3))
