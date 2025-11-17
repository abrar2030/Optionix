import numpy as np
from scipy.stats import norm


class RiskCalculator:
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        return np.percentile(returns, 100 * (1 - confidence))

    @staticmethod
    def expected_shortfall(returns, confidence=0.95):
        var = RiskCalculator.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def margin_requirement(portfolio_value, volatility):
        return portfolio_value * (0.1 + 0.9 * (volatility / 0.3))
