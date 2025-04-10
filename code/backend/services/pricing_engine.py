import numpy as np
from scipy.stats import norm
import sys
sys.path.append('../../quantitative')
from black_scholes import black_scholes

class PricingEngine:
    def calculate_greeks(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)
        return {'delta': delta, 'gamma': gamma, 'vega': vega}

    def monte_carlo_pricing(self, S, K, T, r, sigma, iterations=100000):
        returns = np.random.normal((r-0.5*sigma**2)*T, sigma*np.sqrt(T), iterations)
        ST = S * np.exp(returns)
        payoffs = np.maximum(ST - K, 0)
        return np.exp(-r*T) * np.mean(payoffs)