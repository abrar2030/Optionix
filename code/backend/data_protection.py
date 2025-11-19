"""
Enhanced data protection module for Optionix.
Provides field-level encryption, data masking, and GDPR compliance features.
"""

import base64
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from config import settings
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from models import AuditLog, User
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Base = declarative_base()


class DataClassification(str, Enum):
    """Data classification levels"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PIIType(str, Enum):
    """Types of Personally Identifiable Information"""

    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric"


class EncryptedField:
    """Descriptor for encrypted database fields"""

    def __init__(
        self, classification: DataClassification = DataClassification.CONFIDENTIAL
    ):
        self.classification = classification
        self.encryption_service = None

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        encrypted_value = getattr(obj, self.private_name, None)
        if encrypted_value is None:
            return None

        if self.encryption_service is None:
            self.encryption_service = DataProtectionService()

        try:
            return self.encryption_service.decrypt_field(encrypted_value)
        except Exception as e:
            logger.error(f"Failed to decrypt field {self.name}: {e}")
            return None

    def __set__(self, obj, value):
        if value is None:
            setattr(obj, self.private_name, None)
            return

        if self.encryption_service is None:
            self.encryption_service = DataProtectionService()

        try:
            encrypted_value = self.encryption_service.encrypt_field(value)
            setattr(obj, self.private_name, encrypted_value)
        except Exception as e:
            logger.error(f"Failed to encrypt field {self.name}: {e}")
            setattr(obj, self.private_name, value)  # Fallback to plain text


class DataRetentionPolicy(Base):
    """Data retention policies for different data types"""

    __tablename__ = "data_retention_policies"

    id = Column(Integer, primary_key=True)
    data_type = Column(String(100), nullable=False, unique=True)
    retention_period_days = Column(Integer, nullable=False)
    classification = Column(String(50), nullable=False)
    auto_delete = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataProcessingLog(Base):
    """Log of data processing activities for GDPR compliance"""

    __tablename__ = "data_processing_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    data_subject_id = Column(
        String(100), nullable=False
    )  # Can be user ID or external ID
    processing_activity = Column(String(100), nullable=False)
    data_types = Column(Text, nullable=False)  # JSON array of data types
    legal_basis = Column(String(100), nullable=False)  # GDPR legal basis
    purpose = Column(Text, nullable=False)
    retention_period = Column(Integer, nullable=True)  # Days
    third_party_sharing = Column(Boolean, default=False)
    third_parties = Column(Text, nullable=True)  # JSON array of third parties
    consent_given = Column(Boolean, default=False)
    consent_date = Column(DateTime, nullable=True)
    consent_withdrawn = Column(Boolean, default=False)
    consent_withdrawn_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataSubjectRequest(Base):
    """GDPR data subject requests"""

    __tablename__ = "data_subject_requests"

    id = Column(Integer, primary_key=True)
    request_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, nullable=True)
    email = Column(String(255), nullable=False)
    request_type = Column(
        String(50), nullable=False
    )  # access, rectification, erasure, portability
    status = Column(
        String(50), default="pending"
    )  # pending, processing, completed, rejected
    description = Column(Text, nullable=True)
    verification_method = Column(String(100), nullable=True)
    verification_completed = Column(Boolean, default=False)
    response_data = Column(Text, nullable=True)  # JSON response data
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataProtectionService:
    """Service for data protection, encryption, and GDPR compliance"""

    def __init__(self):
        self._field_encryption_key = None
        self._document_encryption_key = None
        self._load_encryption_keys()

        # PII detection patterns
        self.pii_patterns = {
            PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            PIIType.PHONE: r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            PIIType.SSN: r"\b\d{3}-?\d{2}-?\d{4}\b",
            PIIType.CREDIT_CARD: r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            PIIType.IP_ADDRESS: r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }

    def _load_encryption_keys(self):
        """Load encryption keys for different purposes"""
        try:
            # Field-level encryption key
            field_key_material = (settings.secret_key + "_field").encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"optionix_field_salt",
                iterations=100000,
            )
            field_key = base64.urlsafe_b64encode(kdf.derive(field_key_material))
            self._field_encryption_key = Fernet(field_key)

            # Document-level encryption key
            doc_key_material = (settings.secret_key + "_document").encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"optionix_doc_salt",
                iterations=100000,
            )
            doc_key = base64.urlsafe_b64encode(kdf.derive(doc_key_material))
            self._document_encryption_key = Fernet(doc_key)

        except Exception as e:
            logger.error(f"Failed to load encryption keys: {e}")
            raise

    def encrypt_field(self, data: str) -> str:
        """Encrypt a single field"""
        try:
            encrypted_data = self._field_encryption_key.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Field encryption failed: {e}")
            raise ValueError("Field encryption failed")

    def decrypt_field(self, encrypted_data: str) -> str:
        """Decrypt a single field"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._field_encryption_key.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Field decryption failed: {e}")
            raise ValueError("Field decryption failed")

    def encrypt_document(self, document: Dict[str, Any]) -> str:
        """Encrypt an entire document"""
        try:
            document_json = json.dumps(document, default=str)
            encrypted_data = self._document_encryption_key.encrypt(
                document_json.encode()
            )
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Document encryption failed: {e}")
            raise ValueError("Document encryption failed")

    def decrypt_document(self, encrypted_document: str) -> Dict[str, Any]:
        """Decrypt an entire document"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_document.encode())
            decrypted_data = self._document_encryption_key.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Document decryption failed: {e}")
            raise ValueError("Document decryption failed")

    def mask_pii(self, text: str, pii_type: PIIType, mask_char: str = "*") -> str:
        """Mask PII in text"""
        import re

        if pii_type == PIIType.EMAIL:
            # Mask email: user@domain.com -> u***@domain.com
            return re.sub(r"(\w)[\w._%+-]*(\w)(@[\w.-]+\.\w+)", r"\1***\2\3", text)
        elif pii_type == PIIType.PHONE:
            # Mask phone: 123-456-7890 -> ***-***-7890
            return re.sub(r"(\d{3})[-.]?(\d{3})[-.]?(\d{4})", r"***-***-\3", text)
        elif pii_type == PIIType.SSN:
            # Mask SSN: 123-45-6789 -> ***-**-6789
            return re.sub(r"(\d{3})-?(\d{2})-?(\d{4})", r"***-**-\3", text)
        elif pii_type == PIIType.CREDIT_CARD:
            # Mask credit card: 1234-5678-9012-3456 -> ****-****-****-3456
            return re.sub(
                r"(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})",
                r"****-****-****-\4",
                text,
            )
        else:
            # Generic masking
            if len(text) <= 4:
                return mask_char * len(text)
            return text[:2] + mask_char * (len(text) - 4) + text[-2:]

    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text"""
        import re

        detected_pii = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected_pii.append(
                    {
                        "type": pii_type.value,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.9,  # Simple confidence score
                    }
                )

        return detected_pii

    def sanitize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for logging by masking PII"""
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                # Check if this field contains PII
                pii_detected = self.detect_pii(value)
                if pii_detected:
                    # Mask the PII
                    sanitized_value = value
                    for pii in pii_detected:
                        pii_type = PIIType(pii["type"])
                        sanitized_value = self.mask_pii(sanitized_value, pii_type)
                    sanitized[key] = sanitized_value
                else:
                    sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_for_logging(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_for_logging(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def create_data_processing_log(
        self,
        db: Session,
        data_subject_id: str,
        processing_activity: str,
        data_types: List[str],
        legal_basis: str,
        purpose: str,
        user_id: Optional[int] = None,
        retention_period: Optional[int] = None,
        third_party_sharing: bool = False,
        third_parties: Optional[List[str]] = None,
        consent_given: bool = False,
    ) -> DataProcessingLog:
        """Create a data processing log for GDPR compliance"""

        log_entry = DataProcessingLog(
            user_id=user_id,
            data_subject_id=data_subject_id,
            processing_activity=processing_activity,
            data_types=json.dumps(data_types),
            legal_basis=legal_basis,
            purpose=purpose,
            retention_period=retention_period,
            third_party_sharing=third_party_sharing,
            third_parties=json.dumps(third_parties) if third_parties else None,
            consent_given=consent_given,
            consent_date=datetime.utcnow() if consent_given else None,
        )

        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        return log_entry

    def process_data_subject_request(
        self,
        db: Session,
        request_type: str,
        email: str,
        description: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> DataSubjectRequest:
        """Process GDPR data subject request"""

        request_id = f"DSR_{int(datetime.utcnow().timestamp())}_{secrets.token_hex(4)}"

        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            email=email,
            request_type=request_type,
            description=description,
            status="pending",
        )

        db.add(request)
        db.commit()
        db.refresh(request)

        return request

    def export_user_data(self, db: Session, user_id: int) -> Dict[str, Any]:
        """Export all user data for GDPR data portability"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")

            # Collect all user data
            user_data = {
                "personal_information": {
                    "user_id": user.user_id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "created_at": user.created_at.isoformat(),
                    "updated_at": (
                        user.updated_at.isoformat() if user.updated_at else None
                    ),
                    "kyc_status": user.kyc_status,
                    "is_verified": user.is_verified,
                },
                "accounts": [],
                "trades": [],
                "positions": [],
                "audit_logs": [],
            }

            # Add accounts data
            for account in user.accounts:
                user_data["accounts"].append(
                    {
                        "account_id": account.account_id,
                        "ethereum_address": account.ethereum_address,
                        "account_type": account.account_type,
                        "balance_usd": str(account.balance_usd),
                        "created_at": account.created_at.isoformat(),
                    }
                )

            # Add trades data
            for trade in user.trades:
                user_data["trades"].append(
                    {
                        "trade_id": trade.trade_id,
                        "symbol": trade.symbol,
                        "trade_type": trade.trade_type,
                        "quantity": str(trade.quantity),
                        "price": str(trade.price),
                        "total_value": str(trade.total_value),
                        "status": trade.status,
                        "created_at": trade.created_at.isoformat(),
                    }
                )

            # Add audit logs (limited to user's own actions)
            for log in user.audit_logs:
                user_data["audit_logs"].append(
                    {
                        "action": log.action,
                        "resource_type": log.resource_type,
                        "timestamp": log.timestamp.isoformat(),
                        "status": log.status,
                    }
                )

            return user_data

        except Exception as e:
            logger.error(f"Failed to export user data: {e}")
            raise ValueError(f"Data export failed: {str(e)}")

    def anonymize_user_data(self, db: Session, user_id: int) -> bool:
        """Anonymize user data for GDPR right to erasure"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False

            # Generate anonymous identifier
            anonymous_id = (
                f"anon_{hashlib.sha256(str(user_id).encode()).hexdigest()[:16]}"
            )

            # Anonymize user data
            user.email = f"{anonymous_id}@anonymized.local"
            user.full_name = "Anonymized User"
            user.is_active = False
            user.is_verified = False

            # Update audit logs to remove personal identifiers
            audit_logs = db.query(AuditLog).filter(AuditLog.user_id == user_id).all()
            for log in audit_logs:
                log.ip_address = "0.0.0.0"
                log.user_agent = "Anonymized"
                if log.request_data:
                    log.request_data = json.dumps({"anonymized": True})
                if log.response_data:
                    log.response_data = json.dumps({"anonymized": True})

            db.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to anonymize user data: {e}")
            return False

    def check_data_retention(self, db: Session) -> List[Dict[str, Any]]:
        """Check data retention policies and identify data for deletion"""
        try:
            policies = db.query(DataRetentionPolicy).all()
            deletion_candidates = []

            for policy in policies:
                cutoff_date = datetime.utcnow() - timedelta(
                    days=policy.retention_period_days
                )

                if policy.data_type == "audit_logs":
                    old_logs = (
                        db.query(AuditLog)
                        .filter(AuditLog.timestamp < cutoff_date)
                        .all()
                    )

                    for log in old_logs:
                        deletion_candidates.append(
                            {
                                "type": "audit_log",
                                "id": log.id,
                                "created_at": log.timestamp,
                                "policy": policy.data_type,
                                "auto_delete": policy.auto_delete,
                            }
                        )

                # Add other data types as needed

            return deletion_candidates

        except Exception as e:
            logger.error(f"Failed to check data retention: {e}")
            return []


# Global service instance
data_protection_service = DataProtectionService()
