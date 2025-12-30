# Example: Options Pricing

Demonstrates how to price options using the Black-Scholes model.

## Using Python Library

```python
from code.quantitative.black_scholes import BlackScholesModel, OptionParameters, OptionType

bs_model = BlackScholesModel()

params = OptionParameters(
    spot_price=100.0,
    strike_price=105.0,
    time_to_expiry=0.5,
    risk_free_rate=0.05,
    volatility=0.25,
    option_type=OptionType.CALL
)

result = bs_model.price_option(params)
print(f"Price: ${result.price:.2f}, Delta: {result.delta:.4f}")
```

## Using REST API

```bash
curl -X POST http://localhost:8000/options/price \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"spot_price":100,"strike_price":105,"time_to_expiry":0.5,"risk_free_rate":0.05,"volatility":0.25,"option_type":"call"}'
```
