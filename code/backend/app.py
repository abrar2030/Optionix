from fastapi import FastAPI, HTTPException
from web3 import Web3
import joblib
import numpy as np
import json
import os
import sys

app = FastAPI(title="Optionix API", description="API for options trading platform")

# Setup Web3 connection
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

# Load ABI
try:
    with open(os.path.join(os.path.dirname(__file__), '../blockchain/contracts/FuturesContract.abi.json')) as f:
        Futures_ABI = json.load(f)
except Exception as e:
    print(f"Error loading ABI: {e}")
    Futures_ABI = []

# Load ML model
try:
    model_path = os.path.join(os.path.dirname(__file__), '../ai_models/volatility_model.h5')
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "Welcome to Optionix API", "status": "online"}

@app.post("/predict_volatility")
async def predict_volatility(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Volatility model not available")
    
    try:
        features = np.array([data['open'], data['high'], data['low'], data['volume']]).reshape(1, -1)
        prediction = model.predict(features)
        return {"volatility": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/position_health/{address}")
async def get_position_health(address: str):
    if not w3.is_address(address):
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")
    
    try:
        # In production, this would be a deployed contract address
        contract_address = '0x0000000000000000000000000000000000000000'
        contract = w3.eth.contract(address=contract_address, abi=Futures_ABI)
        position = contract.functions.positions(address).call()
        
        return {
            "address": address,
            "size": position[1],
            "is_long": position[2],
            "entry_price": position[3],
            "liquidation_price": position[3] * 0.9 if position[2] else position[3] * 1.1  # Example logic
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching position: {str(e)}")
