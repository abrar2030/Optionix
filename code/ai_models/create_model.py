import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Create a simple volatility model
X = np.random.rand(100, 4)  # 4 features: open, high, low, volume
y = np.random.rand(100) * 0.2 + 0.1  # Random volatility values between 0.1 and 0.3

# Train a simple model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model
model_path = 'volatility_model.h5'
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
