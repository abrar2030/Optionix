"""
Enhanced security middleware for Optionix backend.
Provides comprehensive security features including headers, rate limiting, and validation.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import redis
from config import settings
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Redis client for rate limiting
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none';"
            ),
            # Strict Transport Security (HTTPS only)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=()"
            ),
            # Server information hiding
            "Server": "Optionix-API",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        response = await call_next(request)

        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value

        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]

        return response


class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with multiple strategies"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.default_limit = settings.rate_limit_requests
        self.default_window = settings.rate_limit_window

        # Different limits for different endpoints
        self.endpoint_limits = {
            "/auth/login": {"requests": 5, "window": 300},  # 5 requests per 5 minutes
            "/auth/register": {"requests": 3, "window": 3600},  # 3 requests per hour
            "/trades": {"requests": 100, "window": 60},  # 100 trades per minute
            "/predict_volatility": {
                "requests": 10,
                "window": 60,
            },  # 10 predictions per minute
        }

        # Global rate limits
        self.global_limits = {
            "requests_per_minute": 1000,
            "requests_per_hour": 10000,
            "requests_per_day": 100000,
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting"""
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        endpoint = request.url.path

        # Check rate limits
        rate_limit_result = await self._check_rate_limits(client_ip, user_id, endpoint)

        if rate_limit_result["exceeded"]:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "retry_after": rate_limit_result["retry_after"],
                    "limit": rate_limit_result["limit"],
                    "remaining": rate_limit_result["remaining"],
                },
                headers={
                    "Retry-After": str(rate_limit_result["retry_after"]),
                    "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                    "X-RateLimit-Remaining": str(rate_limit_result["remaining"]),
                    "X-RateLimit-Reset": str(rate_limit_result["reset_time"]),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit_result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_result["reset_time"])

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available"""
        # This would extract user ID from JWT token if present
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, decode JWT and extract user ID
            return None
        return None

    async def _check_rate_limits(
        self, client_ip: str, user_id: Optional[str], endpoint: str
    ) -> Dict[str, Any]:
        """Check various rate limits"""
        current_time = int(time.time())

        # Determine limits for this endpoint
        if endpoint in self.endpoint_limits:
            limit_config = self.endpoint_limits[endpoint]
            requests_limit = limit_config["requests"]
            window_seconds = limit_config["window"]
        else:
            requests_limit = self.default_limit
            window_seconds = self.default_window

        # Create rate limit key
        identifier = user_id if user_id else client_ip
        rate_key = (
            f"rate_limit:{identifier}:{endpoint}:{current_time // window_seconds}"
        )

        # Check current requests
        current_requests = redis_client.get(rate_key)
        current_requests = int(current_requests) if current_requests else 0

        # Check if limit exceeded
        if current_requests >= requests_limit:
            reset_time = ((current_time // window_seconds) + 1) * window_seconds
            return {
                "exceeded": True,
                "limit": requests_limit,
                "remaining": 0,
                "retry_after": reset_time - current_time,
                "reset_time": reset_time,
            }

        # Increment counter
        pipe = redis_client.pipeline()
        pipe.incr(rate_key)
        pipe.expire(rate_key, window_seconds)
        pipe.execute()

        reset_time = ((current_time // window_seconds) + 1) * window_seconds
        return {
            "exceeded": False,
            "limit": requests_limit,
            "remaining": requests_limit - current_requests - 1,
            "retry_after": 0,
            "reset_time": reset_time,
        }


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.suspicious_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"union\s+select",  # SQL injection
            r"drop\s+table",  # SQL injection
            r"exec\s*\(",  # Code injection
            r"eval\s*\(",  # Code injection
            r"javascript:",  # XSS
            r"vbscript:",  # XSS
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate and sanitize requests"""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "request_too_large",
                    "message": "Request size exceeds maximum allowed",
                },
            )

        # Validate request headers
        validation_result = self._validate_headers(request)
        if not validation_result["valid"]:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "invalid_headers",
                    "message": validation_result["message"],
                },
            )

        # Check for suspicious patterns in URL
        if self._contains_suspicious_patterns(str(request.url)):
            logger.warning(f"Suspicious URL pattern detected: {request.url}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "suspicious_request",
                    "message": "Request contains suspicious patterns",
                },
            )

        # Process request
        response = await call_next(request)

        return response

    def _validate_headers(self, request: Request) -> Dict[str, Any]:
        """Validate request headers"""
        # Check for required headers
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type:
                return {
                    "valid": False,
                    "message": "Content-Type header required for POST/PUT/PATCH requests",
                }

        # Validate User-Agent
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:  # Suspiciously long user agent
            return {"valid": False, "message": "User-Agent header too long"}

        # Check for suspicious headers
        suspicious_headers = ["x-forwarded-host", "x-original-url", "x-rewrite-url"]
        for header in suspicious_headers:
            if header in request.headers:
                logger.warning(f"Suspicious header detected: {header}")

        return {"valid": True, "message": "Headers valid"}

    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        import re

        text_lower = text.lower()

        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive audit logging"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.sensitive_endpoints = [
            "/auth/login",
            "/auth/register",
            "/trades",
            "/positions",
            "/accounts",
            "/predict_volatility",
        ]
        self.log_all_requests = settings.debug

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log requests and responses for audit trail"""
        start_time = time.time()

        # Prepare request data
        request_data = await self._prepare_request_data(request)

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log if needed
        if self._should_log_request(request):
            await self._log_request_response(
                request, response, request_data, process_time
            )

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        return response

    async def _prepare_request_data(self, request: Request) -> Dict[str, Any]:
        """Prepare request data for logging"""
        # Get request body (for POST/PUT requests)
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                # Don't log sensitive data like passwords
                if b"password" in body.lower():
                    body = b"[REDACTED - Contains sensitive data]"
                else:
                    body = body.decode("utf-8", errors="ignore")[:1000]  # Limit size
            except Exception:
                body = "[Could not read body]"

        return {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": body,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
        }

    def _should_log_request(self, request: Request) -> bool:
        """Determine if request should be logged"""
        if self.log_all_requests:
            return True

        # Log sensitive endpoints
        for endpoint in self.sensitive_endpoints:
            if request.url.path.startswith(endpoint):
                return True

        # Log error responses (handled in response logging)
        return False

    async def _log_request_response(
        self,
        request: Request,
        response: Response,
        request_data: Dict[str, Any],
        process_time: float,
    ) -> None:
        """Log request and response data"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "request": request_data,
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "process_time": process_time,
                },
                "security": {
                    "client_ip": request_data["client_ip"],
                    "user_agent": request_data["user_agent"],
                    "endpoint": request.url.path,
                },
            }

            # Log to structured logger
            if response.status_code >= 400:
                logger.warning(f"HTTP {response.status_code}", extra=log_entry)
            else:
                logger.info(f"HTTP {response.status_code}", extra=log_entry)

        except Exception as e:
            logger.error(f"Failed to log request/response: {e}")


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """Middleware for IP whitelisting (optional)"""

    def __init__(self, app: ASGIApp, whitelist: Optional[list] = None):
        super().__init__(app)
        self.whitelist = whitelist or []
        self.enabled = len(self.whitelist) > 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check IP whitelist"""
        if not self.enabled:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if client_ip not in self.whitelist:
            logger.warning(f"Access denied for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "access_denied",
                    "message": "IP address not whitelisted",
                },
            )

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


# Middleware instances
security_headers_middleware = SecurityHeadersMiddleware
advanced_rate_limit_middleware = AdvancedRateLimitMiddleware
request_validation_middleware = RequestValidationMiddleware
audit_logging_middleware = AuditLoggingMiddleware
ip_whitelist_middleware = IPWhitelistMiddleware
