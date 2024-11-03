from fastapi import Request
import time
import logging

logger = logging.getLogger(__name__)


async def log_requests(request: Request, call_next):
    """Middleware to log request details and timing"""
    start_time = time.time()

    response = await call_next(request)

    # Calculate request processing time
    process_time = time.time() - start_time

    # Log request details
    logger.info(
        f"Path: {request.url.path} "
        f"Method: {request.method} "
        f"Processing Time: {process_time:.2f}s "
        f"Status Code: {response.status_code}"
    )

    return response
