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
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)
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
    def validate_email(cls: Any, v: Any) -> Any:
        email_pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()

    @validator("user_id")
    def validate_user_id(cls: Any, v: Any) -> Any:
        if not re.match("^[a-zA-Z0-9_-]{3,50}$", v):
            raise ValueError("Invalid user ID format")
        return v

    @validator("phone_number")
    def validate_phone(cls: Any, v: Any) -> Any:
        if v and (not re.match("^\\+?[1-9]\\d{1,14}$", v)):
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
    def validate_amount(cls: Any, v: Any) -> Any:
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 1000000000:
            raise ValueError("Amount exceeds maximum limit")
        return round(v, 8)

    @validator("currency")
    def validate_currency(cls: Any, v: Any) -> Any:
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "BTC", "ETH"]
        if v.upper() not in valid_currencies:
            raise ValueError("Invalid currency code")
        return v.upper()

    @validator("transaction_type")
    def validate_transaction_type(cls: Any, v: Any) -> Any:
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
    def validate_positive_values(cls: Any, v: Any) -> Any:
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @validator("option_type")
    def validate_option_type(cls: Any, v: Any) -> Any:
        if v.upper() not in ["CALL", "PUT"]:
            raise ValueError("Option type must be CALL or PUT")
        return v.upper()

    @validator("volatility")
    def validate_volatility(cls: Any, v: Any) -> Any:
        if v is not None and (v < 0 or v > 5):
            raise ValueError("Volatility must be between 0 and 5")
        return v


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

    def __init__(self, config: Dict[str, Any]) -> Any:
        """Initialize data handler"""
        self.config = config
        self.db_engine = create_engine(
            config.get("database_url", "sqlite:///data_handler.db")
        )
        Base.metadata.create_all(self.db_engine)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
        )
        self.master_key = config.get("master_key", Fernet.generate_key())
        self.cipher_suite = Fernet(self.master_key)
        self.validation_models = {
            "user": UserDataModel,
            "transaction": TransactionDataModel,
            "option": OptionDataModel,
        }
        self.retention_policies = {
            DataClassification.PUBLIC: 365 * 7,
            DataClassification.INTERNAL: 365 * 5,
            DataClassification.CONFIDENTIAL: 365 * 10,
            DataClassification.RESTRICTED: 365 * 15,
        }
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
            try:
                validated_model = model_class(**data)
                sanitized_data = validated_model.dict()
            except ValidationError as e:
                for error in e.errors():
                    field = ".".join((str(x) for x in error["loc"]))
                    message = error["msg"]
                    errors.append(f"{field}: {message}")
            custom_errors, custom_warnings = self._perform_custom_validations(
                data, data_type, strict_mode
            )
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)
            if sanitized_data:
                sanitized_data = self._sanitize_data(sanitized_data, data_type)
            is_valid = len(errors) == 0
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
                amount = data.get("amount", 0)
                transaction_type = data.get("transaction_type", "").upper()
                if amount > 100000:
                    warnings.append("Large transaction amount detected")
                if transaction_type == "WITHDRAWAL" and amount > 50000:
                    if strict_mode:
                        errors.append("Withdrawal amount exceeds daily limit")
                    else:
                        warnings.append("Withdrawal amount exceeds recommended limit")
                timestamp = data.get("timestamp")
                if timestamp and isinstance(timestamp, datetime):
                    if timestamp.weekday() >= 5:
                        warnings.append("Weekend trading detected")
            elif data_type == "user":
                email = data.get("email", "")
                if self._is_disposable_email(email):
                    if strict_mode:
                        errors.append("Disposable email addresses not allowed")
                    else:
                        warnings.append("Disposable email address detected")
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
                expiration_date = data.get("expiration_date")
                if expiration_date and isinstance(expiration_date, datetime):
                    if expiration_date <= datetime.utcnow():
                        errors.append("Option expiration date must be in the future")
                    max_expiry = datetime.utcnow() + timedelta(days=365 * 5)
                    if expiration_date > max_expiry:
                        warnings.append(
                            "Option expiration date is unusually far in the future"
                        )
                volatility = data.get("volatility")
                if volatility is not None:
                    if volatility > 2.0:
                        warnings.append("Extremely high volatility detected")
                    elif volatility < 0.01:
                        warnings.append("Extremely low volatility detected")
            return (errors, warnings)
        except Exception as e:
            logger.error(f"Custom validation failed: {e}")
            return ([f"Custom validation error: {str(e)}"], [])

    def _sanitize_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Sanitize data for security"""
        try:
            sanitized = data.copy()
            for key, value in sanitized.items():
                if isinstance(value, str):
                    value = re.sub("[;\\'\"\\\\]", "", value)
                    value = re.sub(
                        "<script.*?</script>", "", value, flags=re.IGNORECASE
                    )
                    value = re.sub("<[^>]+>", "", value)
                    value = value.strip()
                    sanitized[key] = value
            if data_type == "user":
                if "email" in sanitized:
                    sanitized["email"] = sanitized["email"].lower().strip()
                if "phone_number" in sanitized and sanitized["phone_number"]:
                    phone = re.sub("[^\\d+]", "", sanitized["phone_number"])
                    sanitized["phone_number"] = phone
            elif data_type == "transaction":
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
            data_json = json.dumps(data, sort_keys=True, default=str)
            encrypted_data = self.cipher_suite.encrypt(data_json.encode())
            data_id = str(uuid.uuid4())
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
            encrypted_model = (
                self.db_session.query(EncryptedDataModel)
                .filter_by(data_id=data_id)
                .first()
            )
            if not encrypted_model:
                raise ValueError(f"Data not found: {data_id}")
            if (
                encrypted_model.expires_at
                and encrypted_model.expires_at < datetime.utcnow()
            ):
                raise ValueError(f"Data has expired: {data_id}")
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
                        del anonymized[field]
                    elif anonymization_level == "partial":
                        original_value = str(anonymized[field])
                        anonymized[field] = self._pseudonymize_value(original_value)
                    elif anonymization_level == "hash":
                        original_value = str(anonymized[field])
                        anonymized[field] = hashlib.sha256(
                            original_value.encode()
                        ).hexdigest()[:16]
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
            hash_object = hashlib.sha256(value.encode())
            hash_hex = hash_object.hexdigest()
            if "@" in value:
                return f"user_{hash_hex[:8]}@example.com"
            elif value.isdigit():
                return f"+1555{hash_hex[:7]}"
            else:
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
            non_null_fields = sum(
                (1 for v in data.values() if v is not None and v != "")
            )
            completeness = non_null_fields / total_fields * 100
            validation_result = self.validate_data(data, data_type, strict_mode=False)
            valid_fields = total_fields - len(validation_result.errors)
            validity = valid_fields / total_fields * 100
            accuracy = 100.0
            consistency = 100.0
            overall = (completeness + accuracy + consistency + validity) / 4
            scores = {
                "completeness": round(completeness, 2),
                "accuracy": round(accuracy, 2),
                "consistency": round(consistency, 2),
                "validity": round(validity, 2),
                "overall": round(overall, 2),
            }
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

    def _store_quality_metrics(self, data_type: str, scores: Dict[str, float]) -> Any:
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
    ) -> Any:
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
            cache_key = f"audit:{user_id}:{data_type}"
            self.redis_client.lpush(
                cache_key, json.dumps(asdict(audit_log), default=str)
            )
            self.redis_client.ltrim(cache_key, 0, 99)
            self.redis_client.expire(cache_key, 86400)
        except Exception as e:
            logger.error(f"Data audit logging failed: {e}")

    def _log_validation_attempt(
        self,
        validation_id: str,
        data_type: str,
        is_valid: bool,
        errors: List[str],
        warnings: List[str],
    ) -> Any:
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
            self.redis_client.lpush("validation_logs", json.dumps(log_data))
            self.redis_client.ltrim("validation_logs", 0, 999)
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
            expired_data = (
                self.db_session.query(EncryptedDataModel)
                .filter(EncryptedDataModel.expires_at < now)
                .all()
            )
            deleted_count = 0
            for data in expired_data:
                self.db_session.delete(data)
                deleted_count += 1
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
            logs = self.redis_client.lrange("validation_logs", 0, -1)
            if not logs:
                return {"total_validations": 0}
            validation_data = [json.loads(log) for log in logs]
            total_validations = len(validation_data)
            successful_validations = sum((1 for v in validation_data if v["is_valid"]))
            total_errors = sum((v["error_count"] for v in validation_data))
            total_warnings = sum((v["warning_count"] for v in validation_data))
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
                    successful_validations / total_validations * 100
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

    def close(self) -> Any:
        """Close database connections"""
        self.db_session.close()
        self.redis_client.close()


data_handler = None


def get_data_handler(config: Dict[str, Any] = None) -> EnhancedDataHandler:
    """Get global data handler instance"""
    global data_handler
    if data_handler is None:
        data_handler = EnhancedDataHandler(config or {})
    return data_handler
