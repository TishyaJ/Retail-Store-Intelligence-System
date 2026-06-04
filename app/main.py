"""
FastAPI application entry point.

- ASGI middleware: extracts/generates X-Correlation-ID as trace_id
- structlog JSON logging with trace_id bound to every log line in request scope
- asyncpg connection pool lifecycle
- All routers registered
"""
from __future__ import annotations

import contextvars
import os
import uuid

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

load_dotenv()

# ---- Configure structlog JSON renderer ----
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), os.environ.get("LOG_LEVEL", "INFO"))
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# ---- FastAPI app ----
app = FastAPI(
    title="Retail Store Intelligence API",
    version="1.0.0",
    description="Purplle Tech Challenge PS3 — Real-time retail analytics from CCTV footage",
)


# ---- ASGI Middleware: trace_id injection ----
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Response:
    trace_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        trace_id=trace_id,
        service="api-server",
        method=request.method,
        path=request.url.path,
    )
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = trace_id
    return response


# ---- Lifecycle: DB pool ----
from .database import close_pool, create_pool  # noqa: E402


@app.on_event("startup")
async def startup() -> None:
    await create_pool()
    logger.info("api_server_started", port=8000)


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_pool()
    logger.info("api_server_stopped")


# ---- Global exception handler ----
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.critical("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": "Internal server error — check logs"},
    )


# ---- Register routers ----
from .ingestion import router as ingest_router   # noqa: E402
from .metrics   import router as metrics_router  # noqa: E402
from .funnel    import router as funnel_router   # noqa: E402
from .anomalies import router as anomaly_router  # noqa: E402
from .health    import router as health_router   # noqa: E402
from .heatmap   import router as heatmap_router  # noqa: E402

app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(anomaly_router)
app.include_router(health_router)
app.include_router(heatmap_router)
