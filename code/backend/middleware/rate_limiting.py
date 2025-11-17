"""
Rate limiting middleware for Optionix backend.
Implements rate limiting to prevent abuse and DDoS attacks.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

import redis
from config import settings
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from security import security_service

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiting service using Redis for distributed rate limiting"""

    def __init__(self):
        """Initialize rate limiter with Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def is_rate_limited(
        self, identifier: str, limit: int = None, window: int = None
    ) -> Dict[str, Any]:
        """
        Check if identifier is rate limited

        Args:
            identifier (str): Unique identifier (IP, user ID, API key)
            limit (int): Request limit (default from settings)
            window (int): Time window in seconds (default from settings)

        Returns:
            Dict[str, Any]: Rate limit status
        """
        if limit is None:
            limit = settings.rate_limit_requests
        if window is None:
            window = settings.rate_limit_window

        if not self.redis_client:
            # Fallback: allow all requests if Redis is unavailable
            logger.warning("Redis unavailable, rate limiting disabled")
            return {
                "limited": False,
                "current_requests": 0,
                "limit": limit,
                "window": window,
                "reset_time": int(time.time()) + window,
                "remaining": limit,
            }

        try:
            current_time = int(time.time())
            window_start = current_time - window

            # Use Redis sorted set for sliding window
            key = f"rate_limit:{identifier}"

            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, window_start)

            # Count current requests
            current_requests = self.redis_client.zcard(key)

            # Check if limit exceeded
            if current_requests >= limit:
                return {
                    "limited": True,
                    "current_requests": current_requests,
                    "limit": limit,
                    "window": window,
                    "reset_time": current_time + window,
                    "remaining": 0,
                }

            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window)

            return {
                "limited": False,
                "current_requests": current_requests + 1,
                "limit": limit,
                "window": window,
                "reset_time": current_time + window,
                "remaining": limit - current_requests - 1,
            }

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fallback: allow request on error
            return {
                "limited": False,
                "current_requests": 0,
                "limit": limit,
                "window": window,
                "reset_time": int(time.time()) + window,
                "remaining": limit,
                "error": str(e),
            }

    def get_client_identifier(self, request: Request) -> str:
        """
        Get unique identifier for rate limiting

        Args:
            request (Request): FastAPI request object

        Returns:
            str: Unique identifier
        """
        # Try to get user ID from token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from auth import verify_token

                token = auth_header.split(" ")[1]
                payload = verify_token(token)
                if payload:
                    return f"user:{payload.get('sub')}"
            except:
                pass

        # Try to get API key
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{security_service.hash_api_key(api_key)[:16]}"

        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware for FastAPI

    Args:
        request (Request): FastAPI request
        call_next: Next middleware/endpoint

    Returns:
        Response with rate limit headers or 429 error
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
        response = await call_next(request)
        return response

    # Get client identifier
    identifier = rate_limiter.get_client_identifier(request)

    # Check rate limit
    rate_limit_status = rate_limiter.is_rate_limited(identifier)

    if rate_limit_status["limited"]:
        # Return 429 Too Many Requests
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "details": {
                    "limit": rate_limit_status["limit"],
                    "window": rate_limit_status["window"],
                    "reset_time": rate_limit_status["reset_time"],
                },
            },
            headers={
                "X-RateLimit-Limit": str(rate_limit_status["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_limit_status["reset_time"]),
                "Retry-After": str(rate_limit_status["window"]),
            },
        )

    # Process request
    response = await call_next(request)

    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(rate_limit_status["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_limit_status["remaining"])
    response.headers["X-RateLimit-Reset"] = str(rate_limit_status["reset_time"])

    return response
