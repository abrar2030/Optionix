"""
Enhanced authentication and authorization module for Optionix.
Provides comprehensive security features including MFA, RBAC, and session management.
"""
import hashlib
import secrets
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import and_
import redis
import json
import logging
from enum import Enum

from config import settings
from database import get_db
from models import User, AuditLog, APIKey
from security import security_service

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis client for session management
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# Security scheme
security = HTTPBearer()


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_MANAGER = "risk_manager"
    API_USER = "api_user"


class Permission(str, Enum):
    """System permissions"""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    
    # Trading
    CREATE_TRADE = "create_trade"
    READ_TRADE = "read_trade"
    UPDATE_TRADE = "update_trade"
    DELETE_TRADE = "delete_trade"
    EXECUTE_TRADE = "execute_trade"
    
    # Positions
    READ_POSITION = "read_position"
    UPDATE_POSITION = "update_position"
    LIQUIDATE_POSITION = "liquidate_position"
    
    # Compliance
    READ_COMPLIANCE = "read_compliance"
    UPDATE_COMPLIANCE = "update_compliance"
    GENERATE_REPORTS = "generate_reports"
    
    # System
    READ_SYSTEM = "read_system"
    UPDATE_SYSTEM = "update_system"
    READ_AUDIT = "read_audit"
    
    # API
    CREATE_API_KEY = "create_api_key"
    READ_API_KEY = "read_api_key"
    DELETE_API_KEY = "delete_api_key"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.CREATE_USER, Permission.READ_USER, Permission.UPDATE_USER, Permission.DELETE_USER,
        Permission.CREATE_TRADE, Permission.READ_TRADE, Permission.UPDATE_TRADE, Permission.DELETE_TRADE,
        Permission.EXECUTE_TRADE, Permission.READ_POSITION, Permission.UPDATE_POSITION,
        Permission.LIQUIDATE_POSITION, Permission.READ_COMPLIANCE, Permission.UPDATE_COMPLIANCE,
        Permission.GENERATE_REPORTS, Permission.READ_SYSTEM, Permission.UPDATE_SYSTEM,
        Permission.READ_AUDIT, Permission.CREATE_API_KEY, Permission.READ_API_KEY,
        Permission.DELETE_API_KEY
    ],
    UserRole.TRADER: [
        Permission.READ_USER, Permission.UPDATE_USER, Permission.CREATE_TRADE,
        Permission.READ_TRADE, Permission.UPDATE_TRADE, Permission.EXECUTE_TRADE,
        Permission.READ_POSITION, Permission.UPDATE_POSITION, Permission.CREATE_API_KEY,
        Permission.READ_API_KEY, Permission.DELETE_API_KEY
    ],
    UserRole.VIEWER: [
        Permission.READ_USER, Permission.READ_TRADE, Permission.READ_POSITION
    ],
    UserRole.COMPLIANCE_OFFICER: [
        Permission.READ_USER, Permission.READ_TRADE, Permission.READ_POSITION,
        Permission.READ_COMPLIANCE, Permission.UPDATE_COMPLIANCE, Permission.GENERATE_REPORTS,
        Permission.READ_AUDIT
    ],
    UserRole.RISK_MANAGER: [
        Permission.READ_USER, Permission.READ_TRADE, Permission.READ_POSITION,
        Permission.UPDATE_POSITION, Permission.LIQUIDATE_POSITION, Permission.READ_COMPLIANCE,
        Permission.GENERATE_REPORTS, Permission.READ_AUDIT
    ],
    UserRole.API_USER: [
        Permission.READ_TRADE, Permission.READ_POSITION, Permission.CREATE_TRADE,
        Permission.EXECUTE_TRADE
    ]
}


class MFAType(str, Enum):
    """Multi-factor authentication types"""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"


class AuthenticationService:
    """Enhanced authentication service with MFA and session management"""
    
    def __init__(self):
        self.session_timeout = timedelta(hours=24)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            if payload.get("type") != token_type:
                return None
            return payload
        except JWTError:
            return None
    
    def create_session(self, user_id: str, user_agent: str, ip_address: str) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            "user_id": user_id,
            "user_agent": user_agent,
            "ip_address": ip_address,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        # Store session in Redis
        redis_client.setex(
            f"session:{session_id}",
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data)
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        session_data = redis_client.get(f"session:{session_id}")
        if session_data:
            return json.loads(session_data)
        return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity"""
        session_data = self.get_session(session_id)
        if session_data:
            session_data["last_activity"] = datetime.utcnow().isoformat()
            redis_client.setex(
                f"session:{session_id}",
                int(self.session_timeout.total_seconds()),
                json.dumps(session_data)
            )
            return True
        return False
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user session"""
        return redis_client.delete(f"session:{session_id}") > 0
    
    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        pattern = f"session:*"
        sessions = redis_client.keys(pattern)
        revoked = 0
        
        for session_key in sessions:
            session_data = redis_client.get(session_key)
            if session_data:
                data = json.loads(session_data)
                if data.get("user_id") == user_id:
                    redis_client.delete(session_key)
                    revoked += 1
        
        return revoked
    
    def check_failed_attempts(self, identifier: str) -> Dict[str, Any]:
        """Check failed login attempts"""
        key = f"failed_attempts:{identifier}"
        attempts = redis_client.get(key)
        
        if attempts is None:
            return {"attempts": 0, "locked": False, "lockout_expires": None}
        
        attempts = int(attempts)
        locked = attempts >= self.max_failed_attempts
        
        if locked:
            ttl = redis_client.ttl(key)
            lockout_expires = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
        else:
            lockout_expires = None
        
        return {
            "attempts": attempts,
            "locked": locked,
            "lockout_expires": lockout_expires
        }
    
    def record_failed_attempt(self, identifier: str) -> None:
        """Record failed login attempt"""
        key = f"failed_attempts:{identifier}"
        current = redis_client.get(key)
        
        if current is None:
            redis_client.setex(key, int(self.lockout_duration.total_seconds()), 1)
        else:
            attempts = int(current) + 1
            if attempts >= self.max_failed_attempts:
                # Lock account
                redis_client.setex(key, int(self.lockout_duration.total_seconds()), attempts)
            else:
                redis_client.incr(key)
    
    def clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed login attempts"""
        redis_client.delete(f"failed_attempts:{identifier}")


class MFAService:
    """Multi-factor authentication service"""
    
    def __init__(self):
        self.issuer_name = "Optionix"
    
    def generate_totp_secret(self) -> str:
        """Generate TOTP secret"""
        return pyotp.random_base32()
    
    def generate_totp_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_totp_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 1 window tolerance
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8-character hex codes
            codes.append(code)
        return codes
    
    def hash_backup_codes(self, codes: List[str]) -> List[str]:
        """Hash backup codes for storage"""
        return [hashlib.sha256(code.encode()).hexdigest() for code in codes]
    
    def verify_backup_code(self, code: str, hashed_codes: List[str]) -> bool:
        """Verify backup code"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        return code_hash in hashed_codes


class RBACService:
    """Role-based access control service"""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def get_user_permissions(self, user_role: UserRole) -> List[Permission]:
        """Get permissions for user role"""
        return self.role_permissions.get(user_role, [])
    
    def check_permission(self, user_role: UserRole, required_permission: Permission) -> bool:
        """Check if user role has required permission"""
        user_permissions = self.get_user_permissions(user_role)
        return required_permission in user_permissions
    
    def require_permission(self, required_permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be used as a dependency in FastAPI
                # Implementation depends on how current user is obtained
                pass
            return wrapper
        return decorator


# Service instances
auth_service = AuthenticationService()
mfa_service = MFAService()
rbac_service = RBACService()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    
    if not auth_service.verify_password(password, user.hashed_password):
        return None
    
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = auth_service.verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        raise credentials_exception
    
    return user


def get_current_verified_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current verified user"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    return current_user


def require_permission(required_permission: Permission):
    """Dependency to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_verified_user)) -> User:
        user_role = UserRole(current_user.role) if hasattr(current_user, 'role') else UserRole.VIEWER
        
        if not rbac_service.check_permission(user_role, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {required_permission.value}"
            )
        
        return current_user
    
    return permission_checker


def log_auth_event(
    db: Session,
    user_id: Optional[int],
    action: str,
    ip_address: str,
    user_agent: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """Log authentication event"""
    try:
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type="authentication",
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            error_message=error_message
        )
        db.add(audit_log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log auth event: {e}")


# Utility functions
def get_password_hash(password: str) -> str:
    """Get password hash"""
    return auth_service.get_password_hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token"""
    return auth_service.create_access_token(data, expires_delta)


def create_refresh_token(data: dict) -> str:
    """Create refresh token"""
    return auth_service.create_refresh_token(data)

