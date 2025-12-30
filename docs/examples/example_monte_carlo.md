# Example: Monte Carlo Simulation

Price exotic options using Monte Carlo simulation.

## Asian Option

```python
from code.quantitative.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator(n_simulations=10000)

asian_price = simulator.price_asian_option(
    spot_price=100.0,
    strike_price=105.0,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.25,
    option_type='call'
)

print(f"Asian Call Price: ${asian_price:.2f}")
```

## REST API

```bash
curl -X POST http://localhost:8000/options/monte-carlo \
  -H "Authorization: Bearer TOKEN" \
  -d '{"option_type":"asian","spot_price":100,"strike_price":105,"time_to_expiry":1,"risk_free_rate":0.05,"volatility":0.25,"call_put":"call","n_simulations":10000}'
```
