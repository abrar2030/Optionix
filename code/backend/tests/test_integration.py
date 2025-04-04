import pytest  
from fastapi.testclient import TestClient  
from app import app  

client = TestClient(app)  

def test_pricing_endpoint():  
    response = client.post("/price_option", json={  
        "S": 150.0,  
        "K": 155.0,  
        "T": 0.5,  
        "r": 0.03,  
        "sigma": 0.25,  
        "option_type": "call"  
    })  
    assert response.status_code == 200  
    assert 5.0 < response.json()["price"] < 10.0  

def test_risk_endpoint():  
    response = client.get("/position_risk/0x123")  
    assert response.status_code == 200  
    assert "delta" in response.json()  