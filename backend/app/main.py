from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import setup_logging, get_logger
import time
import asyncio
from app.config import settings
from app.api.routes import health, metrics, monitoring, strategies, dashboard, risk, signals
from app.api.indicators import router as indicators_router
from app.core.ws_manager import WebSocketManager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

setup_logging()
logger = get_logger("app")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

app = FastAPI(title="TradeSage Expert API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ✨ Añadir CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(monitoring.router, tags=["monitoring"])
app.include_router(indicators_router, tags=["indicators"])
app.include_router(strategies.router, tags=["strategies"])
app.include_router(dashboard.router, tags=["dashboard"])
app.include_router(risk.router, tags=["risk"])
app.include_router(signals.router, tags=["signals"])

# Símbolos a trackear (configurable)
TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

ws_manager = WebSocketManager(symbols=TRACKED_SYMBOLS)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )

    response = await call_next(request)

    duration = time.time() - start_time

    # Log response
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2)
    )

    return response

@app.on_event("startup")
async def startup_event():
    logger.info(
        "application_starting",
        mode=settings.MODE,
        binance_testnet=settings.BINANCE_TESTNET,
        log_level=settings.LOG_LEVEL
    )

    # Validar que las API keys estén configuradas
    if not settings.BINANCE_API_KEY or not settings.BINANCE_API_SECRET:
        logger.error("Binance API keys not configured!")
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")

    logger.info("✅ Configuration validated successfully")

    # Iniciar WebSocket manager en background
    asyncio.create_task(ws_manager.start())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("application_shutting_down")
    await ws_manager.stop()

@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": settings.MODE}

@app.get("/ws/status")
async def get_ws_status():
    return await ws_manager.get_status()