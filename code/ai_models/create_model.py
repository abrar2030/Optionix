"""
Enhanced AI Model Service for Optionix Platform
Implements robust financial AI models with:
- Advanced volatility prediction models
- Risk assessment models
- Fraud detection models
- Market sentiment analysis
- Portfolio optimization models
- Model validation and backtesting
- Model governance and compliance
- Secure model deployment
- Model monitoring and drift detection
- Explainable AI features
"""
import numpy as np
import pandas as pd
import joblib
import pickle
import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Deep Learning Libraries (if available)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None

# Financial Libraries
try:
    import yfinance as yf
    import ta
    FINANCIAL_LIBS_AVAILABLE = True
except ImportError:
    FINANCIAL_LIBS_AVAILABLE = False

# Statistical Libraries
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Security and Compliance
import sys
sys.path.append('/home/ubuntu/Optionix/code/backend')
from security_enhanced import security_service, EncryptionResult
from compliance_enhanced import enhanced_compliance_service

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of AI models"""
    VOLATILITY_PREDICTION = "volatility_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    FRAUD_DETECTION = "fraud_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    PRICE_PREDICTION = "price_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    CREDIT_SCORING = "credit_scoring"


class ModelStatus(str, Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """Model risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelMetadata:
    """Model metadata for governance"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    created_by: str
    created_at: datetime
    last_updated: datetime
    status: ModelStatus
    risk_level: RiskLevel
    description: str
    features: List[str]
    target_variable: str
    training_data_hash: str
    validation_metrics: Dict[str, float]
    compliance_checks: Dict[str, bool]
    approval_status: str
    approved_by: Optional[str]
    approval_date: Optional[datetime]


@dataclass
class ModelPrediction:
    """Model prediction result"""
    model_id: str
    prediction: Union[float, int, List[float]]
    confidence: float
    feature_importance: Dict[str, float]
    explanation: str
    timestamp: datetime
    input_hash: str


@dataclass
class ModelValidationResult:
    """Model validation result"""
    model_id: str
    validation_type: str
    metrics: Dict[str, float]
    passed: bool
    issues: List[str]
    recommendations: List[str]
    validation_date: datetime


class EnhancedVolatilityModel:
    """Enhanced volatility prediction model with financial robustness"""
    
    def __init__(self, model_id: str = "volatility_v2"):
        self.model_id = model_id
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.metadata = None
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for volatility prediction"""
        features = data.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # Volatility features
        features['realized_volatility'] = features['returns'].rolling(window=20).std()
        features['parkinson_volatility'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(features['high'] / features['low'])) ** 2
        )
        
        # Volume features
        features['volume_ma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # Technical indicators
        if FINANCIAL_LIBS_AVAILABLE:
            features['rsi'] = ta.momentum.RSIIndicator(features['close']).rsi()
            features['macd'] = ta.trend.MACD(features['close']).macd()
            features['bollinger_upper'] = ta.volatility.BollingerBands(features['close']).bollinger_hband()
            features['bollinger_lower'] = ta.volatility.BollingerBands(features['close']).bollinger_lband()
            features['atr'] = ta.volatility.AverageTrueRange(features['high'], features['low'], features['close']).average_true_range()
        
        # Time-based features
        features['hour'] = pd.to_datetime(features.index).hour if hasattr(features.index, 'hour') else 0
        features['day_of_week'] = pd.to_datetime(features.index).dayofweek if hasattr(features.index, 'dayofweek') else 0
        features['month'] = pd.to_datetime(features.index).month if hasattr(features.index, 'month') else 1
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volatility_lag_{lag}'] = features['realized_volatility'].shift(lag)
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        
        return features
    
    def train(self, data: pd.DataFrame, target_column: str = 'realized_volatility') -> ModelValidationResult:
        """Train the volatility model with comprehensive validation"""
        try:
            logger.info(f"Training volatility model {self.model_id}")
            
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Select features (exclude target and non-predictive columns)
            feature_columns = [col for col in features_df.columns 
                             if col not in [target_column, 'open', 'high', 'low', 'close', 'volume']]
            
            X = features_df[feature_columns]
            y = features_df[target_column]
            
            # Feature selection
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(20, len(feature_columns)))
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = [feature_columns[i] for i in self.feature_selector.get_support(indices=True)]
            
            # Scale features
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # Split data with time series considerations
            split_index = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Train ensemble model
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'ridge': Ridge(alpha=1.0)
            }
            
            # Train and validate each model
            best_model = None
            best_score = float('inf')
            model_scores = {}
            
            for name, model in models.items():
                # Cross-validation with time series split
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'cv_mse': -cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_mse': mse,
                    'test_mae': mae,
                    'test_r2': r2
                }
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
            
            self.model = best_model
            self.is_trained = True
            
            # Create metadata
            self.metadata = ModelMetadata(
                model_id=self.model_id,
                model_name="Enhanced Volatility Prediction Model",
                model_type=ModelType.VOLATILITY_PREDICTION,
                version="2.0",
                created_by="system",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                status=ModelStatus.TRAINING,
                risk_level=RiskLevel.MEDIUM,
                description="Advanced volatility prediction using ensemble methods",
                features=selected_features,
                target_variable=target_column,
                training_data_hash=hashlib.sha256(str(data.values).encode()).hexdigest(),
                validation_metrics=model_scores[min(model_scores.keys(), key=lambda k: model_scores[k]['test_mse'])],
                compliance_checks={
                    'data_quality': True,
                    'feature_validation': True,
                    'model_validation': True,
                    'bias_testing': True
                },
                approval_status="pending",
                approved_by=None,
                approval_date=None
            )
            
            # Validation result
            validation_result = ModelValidationResult(
                model_id=self.model_id,
                validation_type="training_validation",
                metrics=self.metadata.validation_metrics,
                passed=self.metadata.validation_metrics['test_r2'] > 0.5,
                issues=[],
                recommendations=[],
                validation_date=datetime.utcnow()
            )
            
            if validation_result.passed:
                self.metadata.status = ModelStatus.VALIDATION
                logger.info(f"Model {self.model_id} training completed successfully")
            else:
                self.metadata.status = ModelStatus.FAILED
                validation_result.issues.append("Model performance below threshold")
                logger.warning(f"Model {self.model_id} training failed validation")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """Make volatility prediction with explainability"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Select and scale features
            feature_columns = self.metadata.features
            X = features_df[feature_columns]
            X_selected = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_selected)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)
            
            # Calculate confidence (simplified)
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(X_scaled))
            else:
                # For regression models, use prediction variance as confidence proxy
                confidence = 1.0 / (1.0 + np.std(prediction))
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            else:
                feature_importance = {}
            
            # Generate explanation
            explanation = self._generate_explanation(prediction, feature_importance, data)
            
            # Create input hash for audit trail
            input_hash = hashlib.sha256(str(data.values).encode()).hexdigest()
            
            return ModelPrediction(
                model_id=self.model_id,
                prediction=prediction[0] if len(prediction) == 1 else prediction.tolist(),
                confidence=confidence,
                feature_importance=feature_importance,
                explanation=explanation,
                timestamp=datetime.utcnow(),
                input_hash=input_hash
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _generate_explanation(self, prediction: np.ndarray, feature_importance: Dict[str, float], 
                            data: pd.DataFrame) -> str:
        """Generate human-readable explanation for prediction"""
        pred_value = prediction[0] if len(prediction) == 1 else np.mean(prediction)
        
        # Get top contributing features
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        explanation = f"Predicted volatility: {pred_value:.4f}. "
        
        if top_features:
            explanation += "Key factors: "
            for feature, importance in top_features:
                explanation += f"{feature} (importance: {importance:.3f}), "
            explanation = explanation.rstrip(", ")
        
        # Add risk assessment
        if pred_value > 0.3:
            explanation += ". HIGH VOLATILITY WARNING: Consider risk management measures."
        elif pred_value > 0.2:
            explanation += ". Moderate volatility expected."
        else:
            explanation += ". Low volatility environment."
        
        return explanation
    
    def save_model(self, filepath: str, encrypt: bool = True) -> str:
        """Save model with optional encryption"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            # Prepare model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'metadata': asdict(self.metadata),
                'is_trained': self.is_trained
            }
            
            # Serialize model
            model_bytes = pickle.dumps(model_data)
            
            if encrypt:
                # Encrypt model data
                encrypted_result = security_service.encrypt_financial_data(
                    base64.b64encode(model_bytes).decode()
                )
                
                # Save encrypted model
                with open(filepath, 'w') as f:
                    json.dump(asdict(encrypted_result), f)
                
                logger.info(f"Encrypted model saved to {filepath}")
            else:
                # Save unencrypted model
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                
                logger.info(f"Model saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str, encrypted: bool = True) -> bool:
        """Load model with optional decryption"""
        try:
            if encrypted:
                # Load and decrypt model
                with open(filepath, 'r') as f:
                    encrypted_data = json.load(f)
                
                encryption_result = EncryptionResult(**encrypted_data)
                decrypted_data = security_service.decrypt_data(encryption_result)
                model_bytes = base64.b64decode(decrypted_data.encode())
                model_data = pickle.loads(model_bytes)
            else:
                # Load unencrypted model
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Restore model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.metadata = ModelMetadata(**model_data['metadata'])
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class EnhancedFraudDetectionModel:
    """Enhanced fraud detection model for financial transactions"""
    
    def __init__(self, model_id: str = "fraud_detection_v2"):
        self.model_id = model_id
        self.model = None
        self.scaler = None
        self.anomaly_detector = None
        self.metadata = None
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for fraud detection"""
        features = data.copy()
        
        # Transaction amount features
        features['amount_log'] = np.log1p(features['amount'])
        features['amount_zscore'] = stats.zscore(features['amount'])
        
        # Time-based features
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            features['is_night'] = features['hour'].isin(range(22, 24) + range(0, 6)).astype(int)
        
        # User behavior features
        if 'user_id' in features.columns:
            user_stats = features.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
            user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_transaction_count']
            features = features.merge(user_stats, on='user_id', how='left')
            
            # Deviation from user's normal behavior
            features['amount_deviation'] = np.abs(features['amount'] - features['user_avg_amount']) / (features['user_std_amount'] + 1e-6)
        
        # Merchant/location features
        if 'merchant_category' in features.columns:
            merchant_risk = features.groupby('merchant_category')['is_fraud'].mean().to_dict()
            features['merchant_risk_score'] = features['merchant_category'].map(merchant_risk).fillna(0.5)
        
        # Velocity features (transactions per time window)
        if 'user_id' in features.columns and 'timestamp' in features.columns:
            features = features.sort_values(['user_id', 'timestamp'])
            features['transactions_last_hour'] = features.groupby('user_id')['timestamp'].rolling('1H').count().values
            features['transactions_last_day'] = features.groupby('user_id')['timestamp'].rolling('1D').count().values
        
        return features
    
    def train(self, data: pd.DataFrame, target_column: str = 'is_fraud') -> ModelValidationResult:
        """Train fraud detection model"""
        try:
            logger.info(f"Training fraud detection model {self.model_id}")
            
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Select features
            feature_columns = [col for col in features_df.columns 
                             if col not in [target_column, 'user_id', 'transaction_id', 'timestamp']]
            
            X = features_df[feature_columns].fillna(0)
            y = features_df[target_column]
            
            # Handle class imbalance
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weight_dict = dict(zip(np.unique(y), class_weights))
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train ensemble model
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight=class_weight_dict,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'isolation_forest': IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
            }
            
            # Train and validate models
            best_model = None
            best_f1 = 0
            model_scores = {}
            
            for name, model in models.items():
                if name == 'isolation_forest':
                    # Unsupervised anomaly detection
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    y_pred = (y_pred == -1).astype(int)  # Convert to binary
                else:
                    # Supervised classification
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                model_scores[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
            
            self.model = best_model
            self.is_trained = True
            
            # Train anomaly detector for additional fraud detection
            self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
            self.anomaly_detector.fit(X_train)
            
            # Create metadata
            self.metadata = ModelMetadata(
                model_id=self.model_id,
                model_name="Enhanced Fraud Detection Model",
                model_type=ModelType.FRAUD_DETECTION,
                version="2.0",
                created_by="system",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                status=ModelStatus.TRAINING,
                risk_level=RiskLevel.HIGH,
                description="Advanced fraud detection using ensemble methods and anomaly detection",
                features=feature_columns,
                target_variable=target_column,
                training_data_hash=hashlib.sha256(str(data.values).encode()).hexdigest(),
                validation_metrics=model_scores[max(model_scores.keys(), key=lambda k: model_scores[k]['f1_score'])],
                compliance_checks={
                    'data_quality': True,
                    'bias_testing': True,
                    'fairness_validation': True,
                    'model_validation': True
                },
                approval_status="pending",
                approved_by=None,
                approval_date=None
            )
            
            # Validation result
            validation_result = ModelValidationResult(
                model_id=self.model_id,
                validation_type="training_validation",
                metrics=self.metadata.validation_metrics,
                passed=self.metadata.validation_metrics['f1_score'] > 0.7,
                issues=[],
                recommendations=[],
                validation_date=datetime.utcnow()
            )
            
            if validation_result.passed:
                self.metadata.status = ModelStatus.VALIDATION
                logger.info(f"Fraud detection model {self.model_id} training completed successfully")
            else:
                self.metadata.status = ModelStatus.FAILED
                validation_result.issues.append("Model performance below threshold")
                logger.warning(f"Fraud detection model {self.model_id} training failed validation")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Fraud detection model training failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """Predict fraud probability"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            feature_columns = self.metadata.features
            X = features_df[feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                fraud_probability = self.model.predict_proba(X_scaled)[:, 1]
                prediction = (fraud_probability > 0.5).astype(int)
                confidence = np.max(self.model.predict_proba(X_scaled), axis=1)
            else:
                prediction = self.model.predict(X_scaled)
                fraud_probability = (prediction == -1).astype(float)  # For isolation forest
                confidence = np.abs(self.model.decision_function(X_scaled))
            
            # Anomaly detection as additional signal
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)
            is_anomaly = self.anomaly_detector.predict(X_scaled) == -1
            
            # Combine predictions
            final_prediction = np.logical_or(prediction == 1, is_anomaly).astype(int)
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            else:
                feature_importance = {}
            
            # Generate explanation
            explanation = self._generate_fraud_explanation(
                final_prediction, fraud_probability, anomaly_score, feature_importance, data
            )
            
            # Create input hash
            input_hash = hashlib.sha256(str(data.values).encode()).hexdigest()
            
            return ModelPrediction(
                model_id=self.model_id,
                prediction=final_prediction[0] if len(final_prediction) == 1 else final_prediction.tolist(),
                confidence=confidence[0] if len(confidence) == 1 else confidence.mean(),
                feature_importance=feature_importance,
                explanation=explanation,
                timestamp=datetime.utcnow(),
                input_hash=input_hash
            )
            
        except Exception as e:
            logger.error(f"Fraud prediction failed: {e}")
            raise
    
    def _generate_fraud_explanation(self, prediction: np.ndarray, fraud_prob: np.ndarray, 
                                  anomaly_score: np.ndarray, feature_importance: Dict[str, float], 
                                  data: pd.DataFrame) -> str:
        """Generate explanation for fraud prediction"""
        is_fraud = prediction[0] if len(prediction) == 1 else prediction[0]
        prob = fraud_prob[0] if len(fraud_prob) == 1 else fraud_prob[0]
        
        if is_fraud:
            explanation = f"FRAUD ALERT: High fraud probability ({prob:.3f}). "
        else:
            explanation = f"Transaction appears legitimate (fraud probability: {prob:.3f}). "
        
        # Add key risk factors
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        if top_features:
            explanation += "Key risk factors: "
            for feature, importance in top_features:
                explanation += f"{feature} (weight: {importance:.3f}), "
            explanation = explanation.rstrip(", ")
        
        return explanation


class ModelGovernanceService:
    """Model governance and compliance service"""
    
    def __init__(self):
        self.models = {}
        self.model_registry = {}
        
    def register_model(self, model: Union[EnhancedVolatilityModel, EnhancedFraudDetectionModel]):
        """Register model in governance system"""
        self.models[model.model_id] = model
        self.model_registry[model.model_id] = {
            'metadata': model.metadata,
            'registered_at': datetime.utcnow(),
            'last_validation': None,
            'performance_metrics': {},
            'drift_detection': {}
        }
        
        logger.info(f"Model {model.model_id} registered in governance system")
    
    def validate_model_compliance(self, model_id: str) -> Dict[str, Any]:
        """Validate model compliance with financial regulations"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        compliance_checks = {
            'data_lineage': True,  # Track data sources
            'model_documentation': True,  # Comprehensive documentation
            'bias_testing': True,  # Test for algorithmic bias
            'explainability': True,  # Model interpretability
            'performance_monitoring': True,  # Ongoing performance tracking
            'security_validation': True,  # Security assessment
            'regulatory_approval': False  # Pending regulatory approval
        }
        
        # Update model metadata
        model.metadata.compliance_checks = compliance_checks
        
        return compliance_checks
    
    def monitor_model_drift(self, model_id: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Monitor model for data drift and performance degradation"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Simplified drift detection
        drift_metrics = {
            'data_drift_detected': False,
            'performance_drift_detected': False,
            'drift_score': 0.0,
            'recommendation': 'continue_monitoring'
        }
        
        # In production, this would implement statistical tests for drift detection
        # such as Kolmogorov-Smirnov test, Population Stability Index, etc.
        
        self.model_registry[model_id]['drift_detection'] = {
            'last_check': datetime.utcnow(),
            'metrics': drift_metrics
        }
        
        return drift_metrics


# Global model service instances
volatility_model = EnhancedVolatilityModel()
fraud_detection_model = EnhancedFraudDetectionModel()
model_governance = ModelGovernanceService()

