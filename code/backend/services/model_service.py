"""
ML model service module for Optionix platform.
Handles all machine learning model interactions and predictions.
"""
import joblib
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ModelService:
    """Service for handling ML model predictions"""
    
    def __init__(self):
        """Initialize model service and load the volatility prediction model"""
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load ML model from file"""
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                '../ai_models/volatility_model.h5'
            )
            self.model = joblib.load(model_path)
            logger.info("Volatility model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading volatility model: {e}")
            self.model = None
    
    def is_model_available(self):
        """
        Check if the model is available for predictions
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def predict_volatility(self, market_data):
        """
        Predict volatility based on market data
        
        Args:
            market_data (dict): Market data with open, high, low, volume fields
            
        Returns:
            float: Predicted volatility
            
        Raises:
            ValueError: If model is not available or data is invalid
            Exception: If prediction fails
        """
        if not self.is_model_available():
            raise ValueError("Volatility model not available")
        
        try:
            # Extract and validate required features
            required_fields = ['open', 'high', 'low', 'volume']
            for field in required_fields:
                if field not in market_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Prepare features for prediction
            features = np.array([
                market_data['open'], 
                market_data['high'], 
                market_data['low'], 
                market_data['volume']
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)
            return float(prediction[0])
        except ValueError as e:
            # Re-raise validation errors
            raise e
        except Exception as e:
            logger.error(f"Error during volatility prediction: {e}")
            raise Exception(f"Prediction error: {str(e)}")
