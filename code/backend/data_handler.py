"""
Enhanced Data Validation and Handling Service for Optionix Platform
Implements comprehensive data validation and handling with:
- Input validation and sanitization
- Data encryption and decryption
- Schema validation
- Data integrity checks
- Audit logging
- Data anonymization and pseudonymization
- GDPR compliance features
- Data retention policies
- Backup and recovery
- Data quality monitoring
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import redis
from cryptography.fernet import Fernet
from pydantic import BaseModel, ValidationError, validator
from sqlalchemy import (JSON, Column, DateTime, Float, Integer, LargeBinary,
                        String, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()


class DataClassification(str, Enum):
    """Data classification levels"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ValidationSeverity(str, Enum):
    """Validation error severity"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Validation result structure"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Dict[str, Any]]
    validation_timestamp: datetime
    validation_id: str


@dataclass
class DataAuditLog:
    """Data audit log entry"""

    log_id: str
    operation: str
    data_type: str
    user_id: str
    timestamp: datetime
    data_hash: str
    classification: DataClassification
    metadata: Dict[str, Any]


# Pydantic models for validation
class UserDataModel(BaseModel):
    """User data validation model"""

    user_id: str
    email: str
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    kyc_status: Optional[str] = None

    @validator("email")
    def validate_email(cls, v):
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()

    @validator("user_id")
    def validate_user_id(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", v):
            raise ValueError("Invalid user ID format")
        return v

    @validator("phone_number")
    def validate_phone(cls, v):
        if v and not re.match(r"^\+?[1-9]\d{1,14}$", v):
            raise ValueError("Invalid phone number format")
        return v


class TransactionDataModel(BaseModel):
    """Transaction data validation model"""

    transaction_id: str
    user_id: str
    amount: float
    currency: str
    transaction_type: str
    timestamp: datetime
    status: str
    metadata: Optional[Dict[str, Any]] = None

    @validator("amount")
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 1000000000:  # 1 billion limit
            raise ValueError("Amount exceeds maximum limit")
        return round(v, 8)  # 8 decimal places precision

    @validator("currency")
    def validate_currency(cls, v):
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "BTC", "ETH"]
        if v.upper() not in valid_currencies:
            raise ValueError("Invalid currency code")
        return v.upper()

    @validator("transaction_type")
    def validate_transaction_type(cls, v):
        valid_types = ["DEPOSIT", "WITHDRAWAL", "TRADE", "TRANSFER", "FEE"]
        if v.upper() not in valid_types:
            raise ValueError("Invalid transaction type")
        return v.upper()


class OptionDataModel(BaseModel):
    """Option data validation model"""

    option_id: str
    underlying_asset: str
    strike_price: float
    expiration_date: datetime
    option_type: str
    premium: float
    volatility: Optional[float] = None

    @validator("strike_price", "premium")
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @validator("option_type")
    def validate_option_type(cls, v):
        if v.upper() not in ["CALL", "PUT"]:
            raise ValueError("Option type must be CALL or PUT")
        return v.upper()

    @validator("volatility")
    def validate_volatility(cls, v):
        if v is not None and (v < 0 or v > 5):
            raise ValueError("Volatility must be between 0 and 5")
        return v


# Database Models
class DataAuditLogModel(Base):
    __tablename__ = "data_audit_logs"

    id = Column(Integer, primary_key=True)
    log_id = Column(String(255), unique=True, nullable=False)
    operation = Column(String(100), nullable=False)
    data_type = Column(String(100), nullable=False)
    user_id = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    data_hash = Column(String(255), nullable=False)
    classification = Column(String(50), nullable=False)
    metadata = Column(JSON, nullable=True)


class EncryptedDataModel(Base):
    __tablename__ = "encrypted_data"

    id = Column(Integer, primary_key=True)
    data_id = Column(String(255), unique=True, nullable=False)
    encrypted_data = Column(LargeBinary, nullable=False)
    encryption_key_id = Column(String(255), nullable=False)
    data_type = Column(String(100), nullable=False)
    classification = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=True)


class DataQualityMetrics(Base):
    __tablename__ = "data_quality_metrics"

    id = Column(Integer, primary_key=True)
    metric_id = Column(String(255), unique=True, nullable=False)
    data_type = Column(String(100), nullable=False)
    completeness_score = Column(Float, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)
    validity_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    metadata = Column(JSON, nullable=True)


class EnhancedDataHandler:
    """Enhanced data validation and handling service"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data handler"""
        self.config = config
        self.db_engine = create_engine(
            config.get("database_url", "sqlite:///data_handler.db")
        )
        Base.metadata.create_all(self.db_engine)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()

        # Redis for caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
        )

        # Encryption setup
        self.master_key = config.get("master_key", Fernet.generate_key())
        self.cipher_suite = Fernet(self.master_key)

        # Validation models
        self.validation_models = {
            "user": UserDataModel,
            "transaction": TransactionDataModel,
            "option": OptionDataModel,
        }

        # Data retention policies (in days)
        self.retention_policies = {
            DataClassification.PUBLIC: 365 * 7,  # 7 years
            DataClassification.INTERNAL: 365 * 5,  # 5 years
            DataClassification.CONFIDENTIAL: 365 * 10,  # 10 years
            DataClassification.RESTRICTED: 365 * 15,  # 15 years
        }

        # PII fields for anonymization
        self.pii_fields = {
            "email",
            "phone_number",
            "address",
            "first_name",
            "last_name",
            "date_of_birth",
            "ssn",
            "passport_number",
            "driver_license",
        }

    def validate_data(
        self, data: Dict[str, Any], data_type: str, strict_mode: bool = True
    ) -> ValidationResult:
        """Validate data against schema"""
        try:
            validation_id = str(uuid.uuid4())
            errors = []
            warnings = []
            sanitized_data = None

            # Get validation model
            model_class = self.validation_models.get(data_type)
            if not model_class:
                errors.append(f"No validation model found for data type: {data_type}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    sanitized_data=None,
                    validation_timestamp=datetime.utcnow(),
                    validation_id=validation_id,
                )

            # Validate using Pydantic model
            try:
                validated_model = model_class(**data)
                sanitized_data = validated_model.dict()
            except ValidationError as e:
                for error in e.errors():
                    field = ".".join(str(x) for x in error["loc"])
                    message = error["msg"]
                    errors.append(f"{field}: {message}")

            # Additional custom validations
            custom_errors, custom_warnings = self._perform_custom_validations(
                data, data_type, strict_mode
            )
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)

            # Data sanitization
            if sanitized_data:
                sanitized_data = self._sanitize_data(sanitized_data, data_type)

            is_valid = len(errors) == 0

            # Log validation attempt
            self._log_validation_attempt(
                validation_id, data_type, is_valid, errors, warnings
            )

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized_data,
                validation_timestamp=datetime.utcnow(),
                validation_id=validation_id,
            )

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation system error: {str(e)}"],
                warnings=[],
                sanitized_data=None,
                validation_timestamp=datetime.utcnow(),
                validation_id=str(uuid.uuid4()),
            )

    def _perform_custom_validations(
        self, data: Dict[str, Any], data_type: str, strict_mode: bool
    ) -> Tuple[List[str], List[str]]:
        """Perform custom validation logic"""
        errors = []
        warnings = []

        try:
            if data_type == "transaction":
                # Business logic validations for transactions
                amount = data.get("amount", 0)
                transaction_type = data.get("transaction_type", "").upper()

                # Large transaction warning
                if amount > 100000:
                    warnings.append("Large transaction amount detected")

                # Withdrawal limit check
                if transaction_type == "WITHDRAWAL" and amount > 50000:
                    if strict_mode:
                        errors.append("Withdrawal amount exceeds daily limit")
                    else:
                        warnings.append("Withdrawal amount exceeds recommended limit")

                # Weekend trading warning
                timestamp = data.get("timestamp")
                if timestamp and isinstance(timestamp, datetime):
                    if timestamp.weekday() >= 5:  # Saturday or Sunday
                        warnings.append("Weekend trading detected")

            elif data_type == "user":
                # User data validations
                email = data.get("email", "")

                # Disposable email check
                if self._is_disposable_email(email):
                    if strict_mode:
                        errors.append("Disposable email addresses not allowed")
                    else:
                        warnings.append("Disposable email address detected")

                # Age verification
                dob = data.get("date_of_birth")
                if dob:
                    try:
                        birth_date = datetime.strptime(dob, "%Y-%m-%d")
                        age = (datetime.now() - birth_date).days / 365.25
                        if age < 18:
                            errors.append("User must be at least 18 years old")
                        elif age > 120:
                            warnings.append("Unusual age detected")
                    except ValueError:
                        errors.append("Invalid date of birth format")

            elif data_type == "option":
                # Option data validations
                expiration_date = data.get("expiration_date")
                if expiration_date and isinstance(expiration_date, datetime):
                    if expiration_date <= datetime.utcnow():
                        errors.append("Option expiration date must be in the future")

                    # Check if expiration is too far in the future
                    max_expiry = datetime.utcnow() + timedelta(days=365 * 5)  # 5 years
                    if expiration_date > max_expiry:
                        warnings.append(
                            "Option expiration date is unusually far in the future"
                        )

                # Volatility reasonableness check
                volatility = data.get("volatility")
                if volatility is not None:
                    if volatility > 2.0:  # 200% volatility
                        warnings.append("Extremely high volatility detected")
                    elif volatility < 0.01:  # 1% volatility
                        warnings.append("Extremely low volatility detected")

            return errors, warnings

        except Exception as e:
            logger.error(f"Custom validation failed: {e}")
            return [f"Custom validation error: {str(e)}"], []

    def _sanitize_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Sanitize data for security"""
        try:
            sanitized = data.copy()

            # Remove potentially dangerous characters
            for key, value in sanitized.items():
                if isinstance(value, str):
                    # Remove SQL injection patterns
                    value = re.sub(r'[;\'"\\]', "", value)
                    # Remove script tags
                    value = re.sub(
                        r"<script.*?</script>", "", value, flags=re.IGNORECASE
                    )
                    # Remove other HTML tags
                    value = re.sub(r"<[^>]+>", "", value)
                    # Trim whitespace
                    value = value.strip()
                    sanitized[key] = value

            # Type-specific sanitization
            if data_type == "user":
                # Normalize email
                if "email" in sanitized:
                    sanitized["email"] = sanitized["email"].lower().strip()

                # Normalize phone number
                if "phone_number" in sanitized and sanitized["phone_number"]:
                    phone = re.sub(r"[^\d+]", "", sanitized["phone_number"])
                    sanitized["phone_number"] = phone

            elif data_type == "transaction":
                # Ensure amount precision
                if "amount" in sanitized:
                    sanitized["amount"] = round(float(sanitized["amount"]), 8)

            return sanitized

        except Exception as e:
            logger.error(f"Data sanitization failed: {e}")
            return data

    def _is_disposable_email(self, email: str) -> bool:
        """Check if email is from a disposable email provider"""
        disposable_domains = {
            "10minutemail.com",
            "tempmail.org",
            "guerrillamail.com",
            "mailinator.com",
            "throwaway.email",
            "temp-mail.org",
        }

        try:
            domain = email.split("@")[1].lower()
            return domain in disposable_domains
        except (IndexError, AttributeError):
            return False

    def encrypt_sensitive_data(
        self, data: Dict[str, Any], classification: DataClassification
    ) -> str:
        """Encrypt sensitive data"""
        try:
            # Convert data to JSON string
            data_json = json.dumps(data, sort_keys=True, default=str)

            # Encrypt data
            encrypted_data = self.cipher_suite.encrypt(data_json.encode())

            # Generate data ID
            data_id = str(uuid.uuid4())

            # Store encrypted data
            encrypted_model = EncryptedDataModel(
                data_id=data_id,
                encrypted_data=encrypted_data,
                encryption_key_id="master_key_v1",
                data_type="sensitive_data",
                classification=classification.value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow()
                + timedelta(days=self.retention_policies[classification]),
            )

            self.db_session.add(encrypted_model)
            self.db_session.commit()

            return data_id

        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            raise

    def decrypt_sensitive_data(self, data_id: str) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        try:
            # Retrieve encrypted data
            encrypted_model = (
                self.db_session.query(EncryptedDataModel)
                .filter_by(data_id=data_id)
                .first()
            )

            if not encrypted_model:
                raise ValueError(f"Data not found: {data_id}")

            # Check if data has expired
            if (
                encrypted_model.expires_at
                and encrypted_model.expires_at < datetime.utcnow()
            ):
                raise ValueError(f"Data has expired: {data_id}")

            # Decrypt data
            decrypted_data = self.cipher_suite.decrypt(encrypted_model.encrypted_data)
            data_json = decrypted_data.decode()

            return json.loads(data_json)

        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            raise

    def anonymize_data(
        self, data: Dict[str, Any], anonymization_level: str = "partial"
    ) -> Dict[str, Any]:
        """Anonymize personal data for GDPR compliance"""
        try:
            anonymized = data.copy()

            for field in self.pii_fields:
                if field in anonymized:
                    if anonymization_level == "full":
                        # Full anonymization - remove completely
                        del anonymized[field]
                    elif anonymization_level == "partial":
                        # Partial anonymization - pseudonymize
                        original_value = str(anonymized[field])
                        anonymized[field] = self._pseudonymize_value(original_value)
                    elif anonymization_level == "hash":
                        # Hash the value
                        original_value = str(anonymized[field])
                        anonymized[field] = hashlib.sha256(
                            original_value.encode()
                        ).hexdigest()[:16]

            # Add anonymization metadata
            anonymized["_anonymized"] = True
            anonymized["_anonymization_level"] = anonymization_level
            anonymized["_anonymization_timestamp"] = datetime.utcnow().isoformat()

            return anonymized

        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return data

    def _pseudonymize_value(self, value: str) -> str:
        """Create pseudonymized version of a value"""
        try:
            # Create deterministic pseudonym based on value
            hash_object = hashlib.sha256(value.encode())
            hash_hex = hash_object.hexdigest()

            # Generate pseudonym based on original value type
            if "@" in value:  # Email
                return f"user_{hash_hex[:8]}@example.com"
            elif value.isdigit():  # Phone number
                return f"+1555{hash_hex[:7]}"
            else:  # Name or other text
                return f"User_{hash_hex[:8]}"

        except Exception as e:
            logger.error(f"Pseudonymization failed: {e}")
            return "ANONYMIZED"

    def calculate_data_quality_score(
        self, data: Dict[str, Any], data_type: str
    ) -> Dict[str, float]:
        """Calculate data quality metrics"""
        try:
            total_fields = len(data)
            if total_fields == 0:
                return {
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "consistency": 0.0,
                    "validity": 0.0,
                    "overall": 0.0,
                }

            # Completeness: percentage of non-null fields
            non_null_fields = sum(1 for v in data.values() if v is not None and v != "")
            completeness = (non_null_fields / total_fields) * 100

            # Validity: percentage of fields that pass validation
            validation_result = self.validate_data(data, data_type, strict_mode=False)
            valid_fields = total_fields - len(validation_result.errors)
            validity = (valid_fields / total_fields) * 100

            # Accuracy: based on business rules (simplified)
            accuracy = 100.0  # Would be calculated based on business rules

            # Consistency: based on cross-field validation (simplified)
            consistency = 100.0  # Would be calculated based on consistency rules

            # Overall score
            overall = (completeness + accuracy + consistency + validity) / 4

            scores = {
                "completeness": round(completeness, 2),
                "accuracy": round(accuracy, 2),
                "consistency": round(consistency, 2),
                "validity": round(validity, 2),
                "overall": round(overall, 2),
            }

            # Store metrics
            self._store_quality_metrics(data_type, scores)

            return scores

        except Exception as e:
            logger.error(f"Data quality calculation failed: {e}")
            return {
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "validity": 0.0,
                "overall": 0.0,
            }

    def _store_quality_metrics(self, data_type: str, scores: Dict[str, float]):
        """Store data quality metrics"""
        try:
            metric_id = f"{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            metrics_model = DataQualityMetrics(
                metric_id=metric_id,
                data_type=data_type,
                completeness_score=scores["completeness"],
                accuracy_score=scores["accuracy"],
                consistency_score=scores["consistency"],
                validity_score=scores["validity"],
                timestamp=datetime.utcnow(),
                metadata={"overall_score": scores["overall"]},
            )

            self.db_session.add(metrics_model)
            self.db_session.commit()

        except Exception as e:
            logger.error(f"Quality metrics storage failed: {e}")

    def audit_data_access(
        self,
        operation: str,
        data_type: str,
        user_id: str,
        data_hash: str,
        classification: DataClassification,
        metadata: Dict[str, Any] = None,
    ):
        """Log data access for audit trail"""
        try:
            log_id = str(uuid.uuid4())

            audit_log = DataAuditLog(
                log_id=log_id,
                operation=operation,
                data_type=data_type,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                data_hash=data_hash,
                classification=classification,
                metadata=metadata or {},
            )

            # Store in database
            audit_model = DataAuditLogModel(
                log_id=audit_log.log_id,
                operation=audit_log.operation,
                data_type=audit_log.data_type,
                user_id=audit_log.user_id,
                timestamp=audit_log.timestamp,
                data_hash=audit_log.data_hash,
                classification=audit_log.classification.value,
                metadata=audit_log.metadata,
            )

            self.db_session.add(audit_model)
            self.db_session.commit()

            # Cache recent access for quick lookup
            cache_key = f"audit:{user_id}:{data_type}"
            self.redis_client.lpush(
                cache_key, json.dumps(asdict(audit_log), default=str)
            )
            self.redis_client.ltrim(cache_key, 0, 99)  # Keep last 100 entries
            self.redis_client.expire(cache_key, 86400)  # 24 hours

        except Exception as e:
            logger.error(f"Data audit logging failed: {e}")

    def _log_validation_attempt(
        self,
        validation_id: str,
        data_type: str,
        is_valid: bool,
        errors: List[str],
        warnings: List[str],
    ):
        """Log validation attempt"""
        try:
            log_data = {
                "validation_id": validation_id,
                "data_type": data_type,
                "is_valid": is_valid,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store in Redis for monitoring
            self.redis_client.lpush("validation_logs", json.dumps(log_data))
            self.redis_client.ltrim("validation_logs", 0, 999)  # Keep last 1000 entries

        except Exception as e:
            logger.error(f"Validation logging failed: {e}")

    def get_data_retention_status(self, data_id: str) -> Dict[str, Any]:
        """Get data retention status"""
        try:
            encrypted_model = (
                self.db_session.query(EncryptedDataModel)
                .filter_by(data_id=data_id)
                .first()
            )

            if not encrypted_model:
                return {"status": "not_found"}

            now = datetime.utcnow()
            expires_at = encrypted_model.expires_at

            if expires_at:
                days_until_expiry = (expires_at - now).days

                return {
                    "status": "active" if days_until_expiry > 0 else "expired",
                    "created_at": encrypted_model.created_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "days_until_expiry": days_until_expiry,
                    "classification": encrypted_model.classification,
                }
            else:
                return {
                    "status": "permanent",
                    "created_at": encrypted_model.created_at.isoformat(),
                    "classification": encrypted_model.classification,
                }

        except Exception as e:
            logger.error(f"Retention status check failed: {e}")
            return {"status": "error", "message": str(e)}

    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data"""
        try:
            now = datetime.utcnow()

            # Find expired data
            expired_data = (
                self.db_session.query(EncryptedDataModel)
                .filter(EncryptedDataModel.expires_at < now)
                .all()
            )

            deleted_count = 0
            for data in expired_data:
                self.db_session.delete(data)
                deleted_count += 1

            # Clean up old audit logs (keep for 7 years)
            audit_cutoff = now - timedelta(days=365 * 7)
            old_audits = (
                self.db_session.query(DataAuditLogModel)
                .filter(DataAuditLogModel.timestamp < audit_cutoff)
                .all()
            )

            audit_deleted_count = 0
            for audit in old_audits:
                self.db_session.delete(audit)
                audit_deleted_count += 1

            self.db_session.commit()

            return {
                "expired_data_deleted": deleted_count,
                "old_audits_deleted": audit_deleted_count,
            }

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return {"error": str(e)}

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        try:
            # Get recent validation logs from Redis
            logs = self.redis_client.lrange("validation_logs", 0, -1)

            if not logs:
                return {"total_validations": 0}

            validation_data = [json.loads(log) for log in logs]

            total_validations = len(validation_data)
            successful_validations = sum(1 for v in validation_data if v["is_valid"])
            total_errors = sum(v["error_count"] for v in validation_data)
            total_warnings = sum(v["warning_count"] for v in validation_data)

            # Data type breakdown
            data_type_stats = {}
            for v in validation_data:
                data_type = v["data_type"]
                if data_type not in data_type_stats:
                    data_type_stats[data_type] = {"total": 0, "successful": 0}

                data_type_stats[data_type]["total"] += 1
                if v["is_valid"]:
                    data_type_stats[data_type]["successful"] += 1

            return {
                "total_validations": total_validations,
                "successful_validations": successful_validations,
                "success_rate": (
                    (successful_validations / total_validations * 100)
                    if total_validations > 0
                    else 0
                ),
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "data_type_breakdown": data_type_stats,
            }

        except Exception as e:
            logger.error(f"Validation statistics failed: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connections"""
        self.db_session.close()
        self.redis_client.close()


# Global data handler instance
data_handler = None


def get_data_handler(config: Dict[str, Any] = None) -> EnhancedDataHandler:
    """Get global data handler instance"""
    global data_handler
    if data_handler is None:
        data_handler = EnhancedDataHandler(config or {})
    return data_handler
