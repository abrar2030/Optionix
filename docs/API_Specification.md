# Optionix API Specification

## Overview

This document details the API endpoints provided by the Optionix backend service. The API is built using FastAPI and provides endpoints for options pricing, volatility prediction, and blockchain integration.

```

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Endpoints

### Options Pricing

#### GET /options/pricing

Calculate the price of an option using the Black-Scholes model.

**Request Parameters:**

```json
{
    "underlying_price": float,
    "strike_price": float,
    "time_to_expiry": float,
    "risk_free_rate": float,
    "volatility": float,
    "option_type": "call" | "put"
}
```

**Response:**

```json
{
    "price": float,
    "delta": float,
    "gamma": float,
    "theta": float,
    "vega": float,
    "rho": float
}
```

### Volatility Prediction

#### GET /volatility/predict

Get volatility predictions using AI models.

**Request Parameters:**

```json
{
    "symbol": string,
    "timeframe": "1d" | "1w" | "1m",
    "features": {
        "historical_volatility": float,
        "market_conditions": object,
        "option_chain_data": object
    }
}
```

**Response:**

```json
{
    "predicted_volatility": float,
    "confidence_interval": {
        "lower": float,
        "upper": float
    },
    "timestamp": string
}
```

### Blockchain Integration

#### POST /blockchain/contract

Interact with smart contracts.

**Request Parameters:**

```json
{
    "contract_address": string,
    "function_name": string,
    "parameters": object
}
```

**Response:**

```json
{
    "transaction_hash": string,
    "status": "pending" | "completed" | "failed",
    "gas_used": number
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes and error messages in the following format:

```json
{
    "error": {
        "code": string,
        "message": string,
        "details": object
    }
}
```

Common error codes:

- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting

API requests are limited to:

- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

## WebSocket Endpoints

### /ws/market-data

Real-time market data stream.

**Message Format:**

```json
{
    "type": "price_update" | "order_book_update" | "trade",
    "data": object,
    "timestamp": string
}
```

## Versioning

API version is included in the URL path. Current version: v1
