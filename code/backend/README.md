# Optionix Backend

A comprehensive, secure, and compliant backend for the Optionix options trading platform. This backend implements financial-grade security, regulatory compliance (KYC/AML), and robust trading infrastructure.

## Features

### 🔐 Security & Authentication
- **JWT-based Authentication**: Secure token-based authentication with access and refresh tokens
- **Password Security**: Strong password requirements with bcrypt hashing
- **API Key Management**: Secure API key generation and validation
- **Data Encryption**: AES encryption for sensitive data at rest
- **Rate Limiting**: Distributed rate limiting with Redis to prevent abuse
- **Input Validation**: Comprehensive input sanitization and validation

### 📊 Financial Standards
- **Precise Calculations**: Decimal-based financial calculations for accuracy
- **Risk Management**: Position health monitoring and liquidation price calculations
- **Margin Management**: Sophisticated margin requirement calculations
- **Option Pricing**: Black-Scholes model implementation for option valuation
- **Value at Risk (VaR)**: Portfolio risk assessment capabilities
- **Trading Fees**: Accurate fee calculations with maker/taker differentiation

### 🏛️ Regulatory Compliance
- **KYC (Know Your Customer)**: Complete identity verification workflow
- **AML (Anti-Money Laundering)**: Transaction monitoring and suspicious activity detection
- **Sanctions Screening**: Automated sanctions list checking
- **Audit Trail**: Comprehensive audit logging for all critical operations
- **SAR Generation**: Suspicious Activity Report generation capabilities
- **Transaction Limits**: Configurable daily and position limits

### 🔗 Blockchain Integration
- **Ethereum Integration**: Secure Web3 connectivity with retry logic
- **Smart Contract Interaction**: Position health monitoring from blockchain
- **Transaction Management**: Secure transaction signing and monitoring
- **Gas Optimization**: Intelligent gas estimation and management

### 🤖 Machine Learning
- **Volatility Prediction**: Advanced ML model for market volatility forecasting
- **Model Security**: Model integrity verification and secure loading
- **Feature Engineering**: Sophisticated market data preprocessing
- **Prediction Confidence**: Model confidence scoring and validation

### 📈 Trading Infrastructure
- **Order Management**: Support for market, limit, and stop orders
- **Position Tracking**: Real-time position monitoring and PnL calculation
- **Trade Execution**: Secure and atomic trade processing
- **Portfolio Management**: Multi-account portfolio tracking

## Architecture

### Database Models
- **Users**: User accounts with KYC status and verification
- **Accounts**: Trading accounts linked to Ethereum addresses
- **Positions**: Active trading positions with real-time metrics
- **Trades**: Complete trade history with blockchain integration
- **Audit Logs**: Immutable audit trail for compliance
- **Market Data**: Historical market data for analysis

### Services
- **Authentication Service**: JWT token management and user verification
- **Blockchain Service**: Ethereum network interaction and smart contract calls
- **Model Service**: ML model management and prediction services
- **Financial Service**: Comprehensive financial calculations and risk management
- **Compliance Service**: KYC/AML processing and regulatory compliance
- **Security Service**: Data encryption and security utilities

### Middleware
- **Rate Limiting**: Distributed rate limiting with Redis backend
- **Audit Logging**: Automatic audit trail generation for all operations
- **CORS**: Cross-origin resource sharing configuration
- **Error Handling**: Standardized error responses and logging

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 12+
- Redis 6+
- Ethereum node access (Infura, Alchemy, or local node)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abrar2030/Optionix.git
   cd Optionix/code/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up database**:
   ```bash
   # Create PostgreSQL database
   createdb optionix
   
   # Run migrations (tables will be created automatically on first run)
   python -c "from database import create_tables; create_tables()"
   ```

5. **Start Redis**:
   ```bash
   redis-server
   ```

6. **Run the application**:
   ```bash
   python app.py
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing secret (required) | - |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `ETHEREUM_PROVIDER_URL` | Ethereum node URL | - |
| `FUTURES_CONTRACT_ADDRESS` | Smart contract address | - |
| `MODEL_PATH` | ML model file path | `/app/models/volatility_model.h5` |
| `RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Rate limit window (seconds) | `60` |
| `MAX_POSITION_SIZE` | Maximum position size (USD) | `1000000.0` |
| `MIN_POSITION_SIZE` | Minimum position size (USD) | `100.0` |

### Database Configuration

The backend uses PostgreSQL with SQLAlchemy ORM. Key features:
- **Connection Pooling**: Optimized connection management
- **Encryption at Rest**: Database-level encryption support
- **Indexes**: Optimized queries with strategic indexing
- **Migrations**: Automatic schema management

### Security Configuration

- **JWT Tokens**: 30-minute access tokens, 7-day refresh tokens
- **Password Policy**: Minimum 8 characters with complexity requirements
- **Rate Limiting**: 100 requests per minute per client
- **CORS**: Configurable cross-origin policies
- **Encryption**: AES-256 encryption for sensitive data

## API Documentation

### Authentication Endpoints

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

### Trading Endpoints

#### Create Trade
```http
POST /trades
Authorization: Bearer <token>
Content-Type: application/json

{
  "symbol": "BTC-USD",
  "trade_type": "buy",
  "order_type": "market",
  "quantity": 1.5,
  "price": 50000.00
}
```

#### Get Positions
```http
GET /positions
Authorization: Bearer <token>
```

#### Get Position Health
```http
GET /position_health/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
Authorization: Bearer <token>
```

### Prediction Endpoints

#### Predict Volatility
```http
POST /predict_volatility
Authorization: Bearer <token>
Content-Type: application/json

{
  "open": 50000.0,
  "high": 52000.0,
  "low": 49000.0,
  "volume": 1000000
}
```

### Account Management

#### Create Account
```http
POST /accounts
Authorization: Bearer <token>
Content-Type: application/json

{
  "ethereum_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
  "account_type": "standard"
}
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_comprehensive.py::TestAuthentication -v
pytest tests/test_comprehensive.py::TestSecurity -v
pytest tests/test_comprehensive.py::TestFinancialCalculations -v
pytest tests/test_comprehensive.py::TestCompliance -v
```

### Test Coverage

The test suite covers:
- **Authentication & Authorization**: User registration, login, token validation
- **Security**: Password validation, encryption, input sanitization
- **Financial Calculations**: Margin, liquidation, PnL, option pricing
- **Compliance**: KYC validation, AML monitoring, sanctions screening
- **API Endpoints**: All major endpoints with authentication
- **Rate Limiting**: Abuse prevention and throttling

## Security Considerations

### Production Deployment

1. **Environment Variables**: Never commit secrets to version control
2. **Database Security**: Use encrypted connections and strong passwords
3. **API Keys**: Rotate API keys regularly and use least privilege access
4. **Monitoring**: Implement comprehensive logging and alerting
5. **Backup**: Regular encrypted backups of critical data
6. **Updates**: Keep all dependencies updated for security patches

### Compliance Requirements

1. **Data Retention**: Implement appropriate data retention policies
2. **Audit Trails**: Maintain immutable audit logs for regulatory compliance
3. **KYC/AML**: Ensure proper identity verification and transaction monitoring
4. **Reporting**: Implement required regulatory reporting (SAR, CTR)
5. **Privacy**: Comply with data privacy regulations (GDPR, CCPA)

## Performance Optimization

### Database Optimization
- **Indexing**: Strategic indexes on frequently queried columns
- **Connection Pooling**: Optimized connection management
- **Query Optimization**: Efficient queries with proper joins and filters

### Caching Strategy
- **Redis Caching**: Session data and rate limiting counters
- **Model Caching**: ML model loading and prediction caching
- **API Response Caching**: Cacheable endpoint responses

### Monitoring and Metrics
- **Application Metrics**: Response times, error rates, throughput
- **Business Metrics**: Trading volume, user activity, compliance events
- **Infrastructure Metrics**: Database performance, Redis usage, memory consumption

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check PostgreSQL service status
   - Verify connection string in environment variables
   - Ensure database exists and user has proper permissions

2. **Redis Connection Errors**:
   - Verify Redis service is running
   - Check Redis URL configuration
   - Ensure Redis is accessible from application

3. **Blockchain Connection Issues**:
   - Verify Ethereum provider URL
   - Check API key validity for hosted providers
   - Ensure network connectivity to Ethereum nodes

4. **Model Loading Errors**:
   - Verify model file exists at specified path
   - Check model file integrity
   - Ensure required ML dependencies are installed

### Logging and Debugging

- **Log Levels**: Configure appropriate log levels for different environments
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Error Tracking**: Comprehensive error logging with stack traces
- **Audit Trails**: Complete audit logs for compliance and debugging

## Contributing

1. **Code Standards**: Follow PEP 8 and use type hints
2. **Testing**: Maintain test coverage above 90%
3. **Documentation**: Update documentation for new features
4. **Security**: Security review required for all changes
5. **Compliance**: Ensure regulatory compliance for financial features

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For technical support or questions:
- **Documentation**: Check this README and inline code documentation
- **Issues**: Create GitHub issues for bugs or feature requests
- **Security**: Report security issues privately to the maintainers

---

**⚠️ Important Security Notice**: This backend handles financial data and transactions. Ensure proper security measures are in place before deploying to production, including secure key management, encrypted communications, and regular security audits.

