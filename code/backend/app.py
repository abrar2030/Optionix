from fastapi import FastAPI  
from web3 import Web3  
import joblib  
import numpy as np  

app = FastAPI()  
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))  
model = joblib.load('../ai_models/volatility_model.h5')  

@app.post("/predict_volatility")  
async def predict_volatility(data: dict):  
    features = np.array([data['open'], data['high'], data['low'], data['volume']]).reshape(1, -1)  
    prediction = model.predict(features)  
    return {"volatility": prediction[0][0]}  

@app.get("/position_health/{address}")  
async def get_position_health(address: str):  
    contract = w3.eth.contract(address='0x...', abi=Futures_ABI)  
    position = contract.functions.positions(address).call()  
    return {  
        "size": position[1],  
        "liquidation_price": position[3] * 0.9  # Example logic  
    }  