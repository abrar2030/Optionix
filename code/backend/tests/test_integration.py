import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_integration_root_to_volatility(client: Any, mock_model_service: Any) -> Any:
    """
    Test integration between root endpoint and volatility prediction
    """
    root_response = client.get("/")
    assert root_response.status_code == 200
    assert root_response.json()["status"] == "online"
    mock_model_service.predict_volatility.return_value = 0.25
    test_data = {"open": 100.0, "high": 105.0, "low": 95.0, "volume": 1000000}
    predict_response = client.post("/predict_volatility", json=test_data)
    assert predict_response.status_code == 200
    assert predict_response.json()["volatility"] == 0.25


def test_integration_error_handling(client: Any, mock_blockchain_service: Any) -> Any:
    """
    Test integration of error handling across endpoints
    """
    not_found_response = client.get("/nonexistent_endpoint")
    assert not_found_response.status_code == 404
    invalid_json_response = client.post(
        "/predict_volatility",
        headers={"Content-Type": "application/json"},
        content="invalid json",
    )
    assert invalid_json_response.status_code == 422
    mock_blockchain_service.is_valid_address.return_value = False
    invalid_address_response = client.get("/position_health/invalid-address")
    assert invalid_address_response.status_code == 400
