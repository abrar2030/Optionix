import pytest
from fastapi.testclient import TestClient
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

@pytest.fixture
def client():
    """
    Create a test client for FastAPI application
    """
    return TestClient(app)

@pytest.fixture
def mock_blockchain_service():
    """
    Create a mock for BlockchainService to avoid actual blockchain interactions during tests
    """
    with patch('app.blockchain_service') as mock:
        # Setup mock methods and properties
        mock.is_valid_address.return_value = True
        mock.get_position_health.return_value = {
            "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "size": 100,
            "is_long": True,
            "entry_price": 1000,
            "liquidation_price": 900
        }
        yield mock

@pytest.fixture
def mock_model_service():
    """
    Create a mock for ModelService to avoid loading actual model during tests
    """
    with patch('app.model_service') as mock:
        mock.is_model_available.return_value = True
        mock.predict_volatility.return_value = 0.15  # 15% volatility prediction
        yield mock
