import re
import time
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List, Tuple

from src.logger import get_logger

logger = get_logger("DocForge.DeploySecurity")

# Max upload size configuration: default 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_MIME_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]

def sanitize_filename(filename: str) -> str:
    """Sanitize the uploaded filename to prevent directory traversal or script injection.

    Args:
        filename: Raw filename string from upload client.

    Returns:
        str: Sanitized safe filename.
    """
    # Keep only alphanumeric, dot, underscore, dash
    clean = re.sub(r'[^a-zA-Z0-9_\.\-]', '_', filename)
    # Remove leading dots or duplicate dots
    clean = re.sub(r'\.+', '.', clean).lstrip('.')
    if not clean:
        return "uploaded_document.png"
    return clean

def validate_uploaded_file(content_type: str, file_size: int) -> None:
    """Validate uploaded document specifications.

    Args:
        content_type: MIME type of the uploaded file.
        file_size: Size in bytes of the uploaded file.

    Raises:
        HTTPException: if size is too large or type is unsupported.
    """
    if content_type.lower() not in SUPPORTED_MIME_TYPES:
        logger.warning(f"Rejected unsupported file MIME type: {content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type. Supported formats are: PNG, JPEG, BMP, WEBP."
        )

    if file_size > MAX_FILE_SIZE:
        logger.warning(f"Rejected oversized file upload: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE / (1024 * 1024):.0f}MB."
        )

class SimpleRateLimiterMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per IP address to protect prediction API against DDoS/spam."""

    def __init__(self, app, requests_per_minute: int = 60) -> None:
        """Initialize the rate limiter.

        Args:
            app: Starlette/FastAPI application instance.
            requests_per_minute: Request threshold per IP client per minute.
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        # Dictionary mapping client IP to list of request timestamps
        self.history: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next):
        # Only rate limit the POST prediction endpoint
        if request.method == "POST" and "/predict" in request.url.path:
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            
            # Retrieve client timestamps, filtering out timestamps older than 60 seconds
            client_history = self.history.get(client_ip, [])
            client_history = [t for t in client_history if now - t < 60.0]
            
            if len(client_history) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Too Many Requests",
                        "detail": f"Rate limit exceeded. Limit is {self.requests_per_minute} requests per minute."
                    }
                )
                
            # Record current timestamp
            client_history.append(now)
            self.history[client_ip] = client_history
            
        return await call_next(request)
