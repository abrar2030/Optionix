# Optionix Backend - Startup Instructions

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start the Backend

```bash
# From the code directory
cd /path/to/code
python3 run_backend.py
```

Or using uvicorn directly:

```bash
cd /path/to/code
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Configuration

The backend uses SQLite by default for easy local development.
Edit `.env` to configure:

- Database connection
- Redis (optional, for rate limiting)
- Ethereum provider (optional, for blockchain features)
- ML model paths

## API Documentation

Once started, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Health Check

```bash
curl http://localhost:8000/health
```

## Optional Services

### Redis (for rate limiting)

```bash
docker run -d -p 6379:6379 redis:latest
```

### PostgreSQL (for production)

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:latest
```
