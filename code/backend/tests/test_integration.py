import pytest
from fastapi.testclient import TestClient
import json
import os
import sys
from unittest.mock import patch

# Add the parent directory to sys.path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

def test_integration_root_to_volatility(client, mock_model_service):
    """
    Test integration between root endpoint and volatility prediction
    """
    # First check the API is online
    root_response = client.get("/")
    assert root_response.status_code == 200
    assert root_response.json()["status"] == "online"
    
    # Then make a volatility prediction
    mock_model_service.predict_volatility.return_value = 0.25  # 25% volatility prediction
    
    test_data = {
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "volume": 1000000
    }
    
    predict_response = client.post("/predict_volatility", json=test_data)
    assert predict_response.status_code == 200
    assert predict_response.json()["volatility"] == 0.25

def test_integration_error_handling(client, mock_blockchain_service):
    """
    Test integration of error handling across endpoints
    """
    # Test 404 error
    not_found_response = client.get("/nonexistent_endpoint")
    assert not_found_response.status_code == 404
    
    # Test invalid JSON
    invalid_json_response = client.post(
        "/predict_volatility", 
        headers={"Content-Type": "application/json"},
        content="invalid json"
    )
    assert invalid_json_response.status_code == 422  # FastAPI returns 422 for invalid JSON
    
    # Test invalid address format
    mock_blockchain_service.is_valid_address.return_value = False
    
    invalid_address_response = client.get("/position_health/invalid-address")
    assert invalid_address_response.status_code == 400
