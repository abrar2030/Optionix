import numpy as np  

class MonteCarloSimulator:  
    def __init__(self, S0, mu, sigma, T=1, steps=252):  
        self.S0 = S0  
        self.mu = mu  
        self.sigma = sigma  
        self.T = T  
        self.steps = steps  

    def geometric_brownian_motion(self, n_simulations=10000):  
        dt = self.T/self.steps  
        paths = np.zeros((self.steps+1, n_simulations))  
        paths[0] = self.S0  
        for t in range(1, self.steps+1):  
            rand = np.random.standard_normal(n_simulations)  
            paths[t] = paths[t-1] * np.exp((self.mu - 0.5*self.sigma**2)*dt +  
                                     self.sigma*np.sqrt(dt)*rand)  
        return paths  

    def asian_option_price(self, K, r, simulations=100000):  
        paths = self.geometric_brownian_motion(simulations)  
        avg_price = paths.mean(axis=0)  
        payoff = np.maximum(avg_price - K, 0)  
        return np.exp(-r*self.T) * payoff.mean()  