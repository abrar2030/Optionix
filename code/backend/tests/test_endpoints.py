def test_root_endpoint(client: Any) -> Any:
    """
    Test the root endpoint returns the expected welcome message
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Optionix API", "status": "online"}


def test_predict_volatility_success(client: Any, mock_model_service: Any) -> Any:
    """
    Test successful volatility prediction with valid input data
    """
    test_data = {"open": 100.0, "high": 105.0, "low": 95.0, "volume": 1000000}
    response = client.post("/predict_volatility", json=test_data)
    assert response.status_code == 200
    assert "volatility" in response.json()
    assert isinstance(response.json()["volatility"], float)
    mock_model_service.predict_volatility.assert_called_once()


def test_predict_volatility_missing_data(client: Any) -> Any:
    """
    Test volatility prediction with missing data fields
    """
    test_data = {"open": 100.0, "high": 105.0}
    response = client.post("/predict_volatility", json=test_data)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_volatility_model_unavailable(
    client: Any, mock_model_service: Any
) -> Any:
    """
    Test volatility prediction when model is not available
    """
    mock_model_service.is_model_available.return_value = False
    test_data = {"open": 100.0, "high": 105.0, "low": 95.0, "volume": 1000000}
    response = client.post("/predict_volatility", json=test_data)
    assert response.status_code == 503
    assert "detail" in response.json()
    assert "not available" in response.json()["detail"].lower()


def test_get_position_health_success(client: Any, mock_blockchain_service: Any) -> Any:
    """
    Test successful position health retrieval with valid address
    """
    test_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    response = client.get(f"/position_health/{test_address}")
    assert response.status_code == 200
    assert "address" in response.json()
    assert "size" in response.json()
    assert "is_long" in response.json()
    assert "entry_price" in response.json()
    assert "liquidation_price" in response.json()
    mock_blockchain_service.is_valid_address.assert_called_once_with(test_address)
    mock_blockchain_service.get_position_health.assert_called_once_with(test_address)


def test_get_position_health_invalid_address(
    client: Any, mock_blockchain_service: Any
) -> Any:
    """
    Test position health retrieval with invalid Ethereum address
    """
    mock_blockchain_service.is_valid_address.return_value = False
    test_address = "invalid-ethereum-address"
    response = client.get(f"/position_health/{test_address}")
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "invalid" in response.json()["detail"].lower()


def test_get_position_health_contract_error(
    client: Any, mock_blockchain_service: Any
) -> Any:
    """
    Test position health retrieval when contract call fails
    """
    mock_blockchain_service.get_position_health.side_effect = Exception(
        "Contract error"
    )
    test_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    response = client.get(f"/position_health/{test_address}")
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "contract error" in response.json()["detail"].lower()
