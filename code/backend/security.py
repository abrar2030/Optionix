"""
Security utilities for Optionix backend.
Provides encryption, data protection, and security validation functions.
"""
import hashlib
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
import re
from config import settings

logger = logging.getLogger(__name__)


class SecurityService:
    """Service for encryption, data protection, and security operations"""
    
    def __init__(self):
        """Initialize security service"""
        self._encryption_key = None
        self._load_encryption_key()
    
    def _load_encryption_key(self):
        """Load or generate encryption key"""
        try:
            # In production, this should be loaded from a secure key management service
            key_material = settings.secret_key.encode()
            
            # Derive encryption key from secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'optionix_salt_2024',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key_material))
            self._encryption_key = Fernet(key)
            
        except Exception as e:
            logger.error(f"Failed to load encryption key: {e}")
            raise
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data using Fernet symmetric encryption
        
        Args:
            data (str): Data to encrypt
            
        Returns:
            str: Base64 encoded encrypted data
        """
        try:
            encrypted_data = self._encryption_key.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Data encryption failed")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data (str): Base64 encoded encrypted data
            
        Returns:
            str: Decrypted data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._encryption_key.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Data decryption failed")
    
    def hash_api_key(self, api_key: str) -> str:
        """
        Hash API key for secure storage
        
        Args:
            api_key (str): Plain API key
            
        Returns:
            str: Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_api_key(self) -> Tuple[str, str]:
        """
        Generate a new API key
        
        Returns:
            Tuple[str, str]: (plain_key, hashed_key)
        """
        # Generate random API key
        api_key = 'ok_' + secrets.token_urlsafe(32)
        hashed_key = self.hash_api_key(api_key)
        
        return api_key, hashed_key
    
    def validate_api_key_format(self, api_key: str) -> bool:
        """
        Validate API key format
        
        Args:
            api_key (str): API key to validate
            
        Returns:
            bool: True if format is valid
        """
        # API keys should start with 'ok_' and be followed by 43 characters
        pattern = r'^ok_[A-Za-z0-9_-]{43}$'
        return bool(re.match(pattern, api_key))
    
    def sanitize_input(self, input_data: Any) -> Any:
        """
        Sanitize input data to prevent injection attacks
        
        Args:
            input_data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\';\\]', '', input_data)
            # Limit length to prevent buffer overflow
            sanitized = sanitized[:1000]
            return sanitized.strip()
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        else:
            return input_data
    
    def validate_ethereum_address(self, address: str) -> bool:
        """
        Validate Ethereum address format and checksum
        
        Args:
            address (str): Ethereum address
            
        Returns:
            bool: True if valid
        """
        # Basic format check
        if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
            return False
        
        # Checksum validation (simplified)
        try:
            # Convert to checksum address and compare
            checksum_address = self._to_checksum_address(address)
            return address == checksum_address
        except:
            return False
    
    def _to_checksum_address(self, address: str) -> str:
        """Convert address to checksum format"""
        address = address.lower().replace('0x', '')
        address_hash = hashlib.keccak(address.encode()).hexdigest()
        
        checksum_address = '0x'
        for i, char in enumerate(address):
            if int(address_hash[i], 16) >= 8:
                checksum_address += char.upper()
            else:
                checksum_address += char
        
        return checksum_address
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength
        
        Args:
            password (str): Password to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        issues = []
        score = 0
        
        # Length check
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        else:
            score += 1
        
        # Uppercase check
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        # Lowercase check
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        # Digit check
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        else:
            score += 1
        
        # Special character check
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        else:
            score += 1
        
        # Common password check
        common_passwords = ['password', '123456', 'qwerty', 'admin', 'letmein']
        if password.lower() in common_passwords:
            issues.append("Password is too common")
            score = max(0, score - 2)
        
        # Determine strength
        if score >= 5:
            strength = "strong"
        elif score >= 3:
            strength = "medium"
        else:
            strength = "weak"
        
        return {
            "valid": len(issues) == 0,
            "strength": strength,
            "score": score,
            "issues": issues
        }
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token
        
        Args:
            length (int): Token length
            
        Returns:
            str: Secure token
        """
        return secrets.token_urlsafe(length)
    
    def constant_time_compare(self, a: str, b: str) -> bool:
        """
        Compare two strings in constant time to prevent timing attacks
        
        Args:
            a (str): First string
            b (str): Second string
            
        Returns:
            bool: True if strings are equal
        """
        return secrets.compare_digest(a.encode(), b.encode())
    
    def mask_sensitive_data(self, data: str, mask_char: str = '*', visible_chars: int = 4) -> str:
        """
        Mask sensitive data for logging/display
        
        Args:
            data (str): Data to mask
            mask_char (str): Character to use for masking
            visible_chars (int): Number of characters to keep visible
            
        Returns:
            str: Masked data
        """
        if len(data) <= visible_chars:
            return mask_char * len(data)
        
        return data[:visible_chars] + mask_char * (len(data) - visible_chars)
    
    def validate_request_signature(
        self, 
        payload: str, 
        signature: str, 
        secret: str
    ) -> bool:
        """
        Validate request signature for webhook security
        
        Args:
            payload (str): Request payload
            signature (str): Provided signature
            secret (str): Shared secret
            
        Returns:
            bool: True if signature is valid
        """
        try:
            expected_signature = hashlib.hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return self.constant_time_compare(signature, expected_signature)
        except Exception as e:
            logger.error(f"Signature validation failed: {e}")
            return False
    
    def check_rate_limit_violation(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int,
        current_requests: int
    ) -> Dict[str, Any]:
        """
        Check if rate limit is violated
        
        Args:
            identifier (str): Unique identifier (IP, user ID, etc.)
            limit (int): Request limit
            window_seconds (int): Time window in seconds
            current_requests (int): Current request count
            
        Returns:
            Dict[str, Any]: Rate limit status
        """
        violated = current_requests >= limit
        
        return {
            "violated": violated,
            "current_requests": current_requests,
            "limit": limit,
            "window_seconds": window_seconds,
            "reset_time": datetime.utcnow() + timedelta(seconds=window_seconds),
            "remaining_requests": max(0, limit - current_requests)
        }


# Global security service instance
security_service = SecurityService()

