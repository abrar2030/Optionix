"""
Comprehensive test suite for Optionix backend.
Tests all major components including authentication, trading, compliance, and security.
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
# Import application and dependencies
from app import app
from auth import create_access_token, get_password_hash
from config import settings
from database import Base, get_db
from fastapi.testclient import TestClient
from models import Account, Position, Trade, User
from security import security_service
from services.compliance_service import compliance_service
from services.financial_service import FinancialCalculationService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create test database
Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Test client
client = TestClient(app)


class TestAuthentication:
    """Test authentication and authorization"""

    def test_user_registration(self):
        """Test user registration"""
        user_data = {
            "email": "test@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User",
        }

        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200

        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["full_name"] == user_data["full_name"]
        assert data["is_active"] is True
        assert data["is_verified"] is False

    def test_user_registration_duplicate_email(self):
        """Test registration with duplicate email"""
        user_data = {
            "email": "duplicate@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User",
        }

        # First registration
        response1 = client.post("/auth/register", json=user_data)
        assert response1.status_code == 200

        # Second registration with same email
        response2 = client.post("/auth/register", json=user_data)
        assert response2.status_code == 400
        assert "already registered" in response2.json()["detail"]

    def test_weak_password_rejection(self):
        """Test rejection of weak passwords"""
        user_data = {
            "email": "weak@example.com",
            "password": "weak",
            "full_name": "Test User",
        }

        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 400
        assert "Password validation failed" in response.json()["detail"]

    def test_user_login(self):
        """Test user login"""
        # First register a user
        user_data = {
            "email": "login@example.com",
            "password": "TestPassword123!",
            "full_name": "Login User",
        }
        client.post("/auth/register", json=user_data)

        # Then login
        login_data = {"email": "login@example.com", "password": "TestPassword123!"}

        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_invalid_login(self):
        """Test login with invalid credentials"""
        login_data = {"email": "nonexistent@example.com", "password": "wrongpassword"}

        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]


class TestSecurity:
    """Test security utilities and validation"""

    def test_password_strength_validation(self):
        """Test password strength validation"""
        # Strong password
        result = security_service.validate_password_strength("StrongPass123!")
        assert result["valid"] is True
        assert result["strength"] == "strong"

        # Weak password
        result = security_service.validate_password_strength("weak")
        assert result["valid"] is False
        assert result["strength"] == "weak"

    def test_ethereum_address_validation(self):
        """Test Ethereum address validation"""
        # Valid address
        valid_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        assert security_service.validate_ethereum_address(valid_address) is True

        # Invalid address
        invalid_address = "0xinvalid"
        assert security_service.validate_ethereum_address(invalid_address) is False

    def test_api_key_generation(self):
        """Test API key generation"""
        plain_key, hashed_key = security_service.generate_api_key()

        assert plain_key.startswith("ok_")
        assert len(plain_key) == 46  # 'ok_' + 43 characters
        assert security_service.validate_api_key_format(plain_key) is True
        assert security_service.hash_api_key(plain_key) == hashed_key

    def test_data_encryption(self):
        """Test data encryption and decryption"""
        original_data = "sensitive information"

        encrypted = security_service.encrypt_sensitive_data(original_data)
        assert encrypted != original_data

        decrypted = security_service.decrypt_sensitive_data(encrypted)
        assert decrypted == original_data

    def test_input_sanitization(self):
        """Test input sanitization"""
        malicious_input = "<script>alert('xss')</script>"
        sanitized = security_service.sanitize_input(malicious_input)
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # Content preserved, tags removed


class TestFinancialCalculations:
    """Test financial calculation service"""

    def setUp(self):
        """Set up test fixtures"""
        self.financial_service = FinancialCalculationService()

    def test_liquidation_price_calculation(self):
        """Test liquidation price calculation"""
        entry_price = Decimal("50000")
        position_size = Decimal("1.0")
        initial_margin = Decimal("5000")

        # Long position
        liquidation_price = self.financial_service.calculate_liquidation_price(
            entry_price, position_size, True, initial_margin
        )
        assert liquidation_price < entry_price

        # Short position
        liquidation_price = self.financial_service.calculate_liquidation_price(
            entry_price, position_size, False, initial_margin
        )
        assert liquidation_price > entry_price

    def test_margin_requirement_calculation(self):
        """Test margin requirement calculation"""
        position_value = Decimal("100000")
        leverage = Decimal("10")

        margin = self.financial_service.calculate_margin_requirement(
            position_value, leverage
        )
        assert margin == Decimal("10000")  # 100000 / 10

    def test_unrealized_pnl_calculation(self):
        """Test unrealized PnL calculation"""
        entry_price = Decimal("50000")
        current_price = Decimal("55000")
        position_size = Decimal("1.0")

        # Long position with profit
        pnl = self.financial_service.calculate_unrealized_pnl(
            entry_price, current_price, position_size, True
        )
        assert pnl == Decimal("5000")  # (55000 - 50000) * 1.0

        # Short position with loss
        pnl = self.financial_service.calculate_unrealized_pnl(
            entry_price, current_price, position_size, False
        )
        assert pnl == Decimal("-5000")  # (50000 - 55000) * 1.0

    def test_trading_fees_calculation(self):
        """Test trading fees calculation"""
        trade_value = Decimal("10000")

        # Taker fee
        fee = self.financial_service.calculate_trading_fees(trade_value, is_maker=False)
        assert fee == Decimal("10.00")  # 0.1% of 10000

        # Maker fee (lower)
        fee = self.financial_service.calculate_trading_fees(trade_value, is_maker=True)
        assert fee == Decimal("5.00")  # 0.05% of 10000

    def test_position_limits_validation(self):
        """Test position limits validation"""
        position_size = Decimal("1.0")
        position_value = Decimal("50000")
        account_balance = Decimal("10000")

        result = self.financial_service.validate_position_limits(
            position_size, position_value, account_balance
        )

        assert "valid" in result
        assert "violations" in result
        assert "required_margin" in result


class TestCompliance:
    """Test compliance and regulatory features"""

    def test_kyc_data_validation(self):
        """Test KYC data validation"""
        valid_kyc_data = {
            "full_name": "John Doe",
            "date_of_birth": "1990-01-01",
            "nationality": "US",
            "address": "123 Main St, City, State",
            "document_type": "passport",
            "document_number": "A12345678",
            "document_expiry": "2030-01-01",
        }

        result = compliance_service.validate_kyc_data(valid_kyc_data)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_kyc_data_validation_errors(self):
        """Test KYC data validation with errors"""
        invalid_kyc_data = {
            "full_name": "J",  # Too short
            "date_of_birth": "2010-01-01",  # Too young
            "nationality": "XX",  # Invalid country
            "document_type": "invalid",  # Invalid document type
            "document_expiry": "2020-01-01",  # Expired
        }

        result = compliance_service.validate_kyc_data(invalid_kyc_data)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_sanctions_list_check(self):
        """Test sanctions list checking"""
        # Clean name
        result = compliance_service.check_sanctions_list("Clean User", "US")
        assert result["sanctioned"] is False

        # Sanctioned name (from mock list)
        result = compliance_service.check_sanctions_list("John Doe", "US")
        assert result["sanctioned"] is True
        assert result["name_match"] is True

    def test_transaction_compliance_check(self):
        """Test transaction compliance checking"""
        # Create test database session
        db = TestingSessionLocal()

        try:
            # Create test user
            user = User(
                email="compliance@test.com",
                hashed_password=get_password_hash("password"),
                full_name="Compliance User",
                kyc_status="approved",
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            # Test compliant transaction
            trade_data = {
                "symbol": "BTC-USD",
                "trade_type": "buy",
                "quantity": 1.0,
                "total_value": 5000.0,
            }

            result = compliance_service.check_transaction_compliance(
                trade_data, user.id, db
            )
            assert result["compliant"] is True

            # Test non-compliant transaction (too large)
            large_trade_data = {
                "symbol": "BTC-USD",
                "trade_type": "buy",
                "quantity": 100.0,
                "total_value": 100000.0,  # Exceeds daily limit
            }

            result = compliance_service.check_transaction_compliance(
                large_trade_data, user.id, db
            )
            assert result["compliant"] is False
            assert len(result["violations"]) > 0

        finally:
            db.close()


class TestAPIEndpoints:
    """Test API endpoints with authentication"""

    def get_auth_headers(self):
        """Get authentication headers for testing"""
        # Register and login a test user
        user_data = {
            "email": "api@test.com",
            "password": "TestPassword123!",
            "full_name": "API Test User",
        }
        client.post("/auth/register", json=user_data)

        login_response = client.post(
            "/auth/login",
            json={"email": "api@test.com", "password": "TestPassword123!"},
        )

        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without authentication"""
        response = client.get("/positions")
        assert response.status_code == 401

    def test_protected_endpoint_with_auth(self):
        """Test accessing protected endpoint with authentication"""
        headers = self.get_auth_headers()
        response = client.get("/positions", headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_volatility_prediction(self):
        """Test volatility prediction endpoint"""
        headers = self.get_auth_headers()

        market_data = {
            "open": 50000.0,
            "high": 52000.0,
            "low": 49000.0,
            "volume": 1000000,
        }

        response = client.post("/predict_volatility", json=market_data, headers=headers)
        # Note: This might fail if model is not available, which is expected in test environment
        assert response.status_code in [200, 503]

    def test_account_creation(self):
        """Test account creation endpoint"""
        headers = self.get_auth_headers()

        account_data = {
            "ethereum_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "account_type": "standard",
        }

        response = client.post("/accounts", json=account_data, headers=headers)
        assert response.status_code == 200

        data = response.json()
        assert data["ethereum_address"] == account_data["ethereum_address"]
        assert data["account_type"] == account_data["account_type"]


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limiting(self):
        """Test rate limiting enforcement"""
        # Make multiple requests quickly
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response)

        # All requests should succeed for health endpoint (excluded from rate limiting)
        assert all(r.status_code == 200 for r in responses)

        # Check rate limit headers are present
        last_response = responses[-1]
        assert "X-RateLimit-Limit" in last_response.headers
        assert "X-RateLimit-Remaining" in last_response.headers


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
