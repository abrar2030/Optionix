"""
Enhanced ML model service module for Optionix platform.
Handles machine learning model interactions with robust validation and security.
"""
import joblib
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
import pickle
import hashlib
from datetime import datetime, timedelta
from config import settings
from models import MarketData, AuditLog
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)


class ModelService:
    """Enhanced service for handling ML model predictions with security and validation"""
    
    def __init__(self):
        """Initialize model service and load the volatility prediction model"""
        self.model = None
        self.model_metadata = {}
        self.feature_scaler = None
        self.model_hash = None
        self._load_model()
        self._load_feature_scaler()
    
    def _load_model(self):
        """Load ML model from file with integrity verification"""
        try:
            model_path = settings.model_path
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                self.model = None
                return
            
            # Verify model file integrity
            if not self._verify_model_integrity(model_path):
                logger.error("Model file integrity check failed")
                self.model = None
                return
            
            # Load the model
            if model_path.endswith('.h5'):
                # TensorFlow/Keras model
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                except ImportError:
                    logger.error("TensorFlow not available for .h5 model")
                    self.model = None
                    return
            else:
                # Scikit-learn or other joblib-compatible model
                self.model = joblib.load(model_path)
            
            # Load model metadata
            self._load_model_metadata(model_path)
            
            logger.info(f"Volatility model loaded successfully (version: {self.model_metadata.get('version', 'unknown')})")
            
        except Exception as e:
            logger.error(f"Error loading volatility model: {e}")
            self.model = None
    
    def _load_feature_scaler(self):
        """Load feature scaler for input normalization"""
        try:
            scaler_path = os.path.join(os.path.dirname(settings.model_path), 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            else:
                logger.warning("Feature scaler not found, using raw features")
                self.feature_scaler = None
        except Exception as e:
            logger.error(f"Error loading feature scaler: {e}")
            self.feature_scaler = None
    
    def _verify_model_integrity(self, model_path: str) -> bool:
        """
        Verify model file integrity using hash comparison
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            bool: True if integrity check passes
        """
        try:
            # Calculate file hash
            with open(model_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            self.model_hash = file_hash
            
            # In production, compare with known good hash
            # For now, just store the hash for logging
            logger.info(f"Model file hash: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying model integrity: {e}")
            return False
    
    def _load_model_metadata(self, model_path: str):
        """Load model metadata from accompanying JSON file"""
        try:
            metadata_path = model_path.replace('.h5', '_metadata.json').replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            else:
                # Default metadata
                self.model_metadata = {
                    "version": settings.model_version,
                    "created_at": datetime.utcnow().isoformat(),
                    "features": ["open", "high", "low", "volume"],
                    "target": "volatility",
                    "model_type": "unknown"
                }
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            self.model_metadata = {}
    
    def is_model_available(self) -> bool:
        """
        Check if the model is available for predictions
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "available": self.is_model_available(),
            "version": self.model_metadata.get("version", "unknown"),
            "model_type": self.model_metadata.get("model_type", "unknown"),
            "features": self.model_metadata.get("features", []),
            "hash": self.model_hash,
            "last_loaded": datetime.utcnow().isoformat()
        }
    
    def validate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of market data input
        
        Args:
            market_data (Dict[str, Any]): Raw market data
            
        Returns:
            Dict[str, Any]: Validated and cleaned market data
            
        Raises:
            ValueError: If validation fails
        """
        # Required fields
        required_fields = ['open', 'high', 'low', 'volume']
        for field in required_fields:
            if field not in market_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert to appropriate types and validate ranges
        validated_data = {}
        
        try:
            # Price fields must be positive
            for price_field in ['open', 'high', 'low']:
                value = float(market_data[price_field])
                if value <= 0:
                    raise ValueError(f"{price_field} must be positive")
                if value > 1000000:  # Sanity check for extremely high prices
                    raise ValueError(f"{price_field} value seems unrealistic: {value}")
                validated_data[price_field] = value
            
            # Volume must be positive integer
            volume = int(market_data['volume'])
            if volume <= 0:
                raise ValueError("Volume must be positive")
            if volume > 1e12:  # Sanity check for extremely high volume
                raise ValueError(f"Volume value seems unrealistic: {volume}")
            validated_data['volume'] = volume
            
            # Validate price relationships
            if validated_data['high'] < validated_data['low']:
                raise ValueError("High price cannot be less than low price")
            
            if not (validated_data['low'] <= validated_data['open'] <= validated_data['high']):
                logger.warning("Open price is outside high-low range, which is unusual but allowed")
            
            # Calculate additional features
            validated_data['price_range'] = validated_data['high'] - validated_data['low']
            validated_data['price_change'] = abs(validated_data['high'] - validated_data['low']) / validated_data['open']
            
            return validated_data
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid market data: {str(e)}")
    
    def preprocess_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess market data into model features
        
        Args:
            market_data (Dict[str, Any]): Validated market data
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        # Extract base features
        features = [
            market_data['open'],
            market_data['high'],
            market_data['low'],
            market_data['volume']
        ]
        
        # Add derived features
        features.extend([
            market_data['price_range'],
            market_data['price_change'],
            market_data['volume'] / 1000000,  # Volume in millions
            (market_data['high'] + market_data['low']) / 2,  # Mid price
        ])
        
        # Convert to numpy array
        feature_array = np.array(features).reshape(1, -1)
        
        # Apply feature scaling if available
        if self.feature_scaler is not None:
            try:
                feature_array = self.feature_scaler.transform(feature_array)
            except Exception as e:
                logger.warning(f"Feature scaling failed, using raw features: {e}")
        
        return feature_array
    
    def predict_volatility(self, market_data: Dict[str, Any], db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Predict volatility based on market data with comprehensive validation
        
        Args:
            market_data (Dict[str, Any]): Market data with open, high, low, volume fields
            db (Optional[Session]): Database session for logging
            
        Returns:
            Dict[str, Any]: Prediction results with volatility and metadata
            
        Raises:
            ValueError: If model is not available or data is invalid
            Exception: If prediction fails
        """
        if not self.is_model_available():
            raise ValueError("Volatility model not available")
        
        start_time = datetime.utcnow()
        
        try:
            # Validate input data
            validated_data = self.validate_market_data(market_data)
            
            # Preprocess features
            features = self.preprocess_features(validated_data)
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                # Scikit-learn style model
                prediction = self.model.predict(features)
                volatility = float(prediction[0])
                
                # Get prediction confidence if available
                confidence = None
                if hasattr(self.model, 'predict_proba'):
                    try:
                        proba = self.model.predict_proba(features)
                        confidence = float(np.max(proba))
                    except:
                        pass
                        
            elif hasattr(self.model, '__call__'):
                # TensorFlow/Keras model
                prediction = self.model(features)
                volatility = float(prediction.numpy()[0][0])
                confidence = None
            else:
                raise ValueError("Unknown model type")
            
            # Validate prediction output
            if volatility < 0:
                logger.warning(f"Negative volatility predicted: {volatility}, setting to 0")
                volatility = 0.0
            elif volatility > 5.0:  # 500% volatility seems unrealistic
                logger.warning(f"Extremely high volatility predicted: {volatility}")
            
            # Calculate prediction time
            prediction_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Prepare result
            result = {
                "volatility": volatility,
                "confidence": confidence,
                "model_version": self.model_metadata.get("version", "unknown"),
                "prediction_time_ms": prediction_time * 1000,
                "features_used": list(validated_data.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log prediction for audit trail
            if db:
                self._log_prediction(db, validated_data, result, "success")
            
            return result
            
        except ValueError as e:
            # Log validation errors
            if db:
                self._log_prediction(db, market_data, {"error": str(e)}, "validation_error")
            raise e
        except Exception as e:
            # Log prediction errors
            if db:
                self._log_prediction(db, market_data, {"error": str(e)}, "prediction_error")
            logger.error(f"Error during volatility prediction: {e}")
            raise Exception(f"Prediction error: {str(e)}")
    
    def _log_prediction(
        self, 
        db: Session, 
        input_data: Dict[str, Any], 
        result: Dict[str, Any], 
        status: str
    ):
        """Log prediction for audit trail"""
        try:
            audit_log = AuditLog(
                action="volatility_prediction",
                resource_type="model",
                request_data=json.dumps(input_data),
                response_data=json.dumps(result),
                status=status
            )
            db.add(audit_log)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            db.rollback()
    
    def get_historical_predictions(
        self, 
        db: Session, 
        limit: int = 100,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical volatility predictions
        
        Args:
            db (Session): Database session
            limit (int): Maximum number of records to return
            symbol (Optional[str]): Filter by symbol
            
        Returns:
            List[Dict[str, Any]]: Historical predictions
        """
        try:
            query = db.query(MarketData).order_by(MarketData.timestamp.desc())
            
            if symbol:
                query = query.filter(MarketData.symbol == symbol)
            
            records = query.limit(limit).all()
            
            return [
                {
                    "symbol": record.symbol,
                    "timestamp": record.timestamp.isoformat(),
                    "open_price": float(record.open_price),
                    "high_price": float(record.high_price),
                    "low_price": float(record.low_price),
                    "close_price": float(record.close_price),
                    "volume": float(record.volume),
                    "volatility": float(record.volatility) if record.volatility else None
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Error fetching historical predictions: {e}")
            return []
    
    def update_model(self, new_model_path: str) -> bool:
        """
        Update the model with a new version
        
        Args:
            new_model_path (str): Path to new model file
            
        Returns:
            bool: True if update successful
        """
        try:
            # Backup current model info
            old_model_info = self.get_model_info()
            
            # Load new model
            old_model = self.model
            old_metadata = self.model_metadata
            
            # Temporarily update path and reload
            original_path = settings.model_path
            settings.model_path = new_model_path
            
            self._load_model()
            
            if self.is_model_available():
                logger.info(f"Model updated successfully from {old_model_info.get('version')} to {self.model_metadata.get('version')}")
                return True
            else:
                # Rollback on failure
                self.model = old_model
                self.model_metadata = old_metadata
                settings.model_path = original_path
                logger.error("Model update failed, rolled back to previous version")
                return False
                
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False

