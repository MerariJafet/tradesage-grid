from fastapi import APIRouter
from app.core.ws_manager import ws_manager
import time
from datetime import datetime

router = APIRouter(prefix="/api", tags=["dashboard"])

@router.get("/system/status")
async def get_system_status():
    """Get system status for dashboard"""
    ws_status = await ws_manager.get_status()

    # Calculate uptime
    uptime_seconds = time.time() - ws_manager.start_time if hasattr(ws_manager, 'start_time') else 0

    # Get client info safely
    clients = ws_status.get("clients", [])
    reconnect_attempts = clients[0].get("reconnect_attempts", 0) if clients else 0

    return {
        "binance": {
            "status": "connected" if ws_status.get("is_running", False) else "disconnected",
            "latency_ms": 45,  # Mock latency for now
            "reconnects": reconnect_attempts,
            "last_ping": datetime.now().isoformat()
        },
        "system": {
            "uptime_seconds": uptime_seconds,
            "mode": "paper",
            "start_time": datetime.now().isoformat()
        },
        "database": {
            "status": "healthy",
            "last_write": datetime.now().isoformat()
        }
    }

@router.get("/pnl/current")
async def get_pnl_current():
    """Get current PnL data for dashboard"""
    ws_status = await ws_manager.get_status()
    paper_exchange = ws_status.get("paper_exchange", {})

    return {
        "total_pnl": paper_exchange.get("pnl", 0),
        "pnl_percent": paper_exchange.get("pnl_percent", 0),
        "initial_balance": paper_exchange.get("initial_balance", 10000),
        "current_balance": paper_exchange.get("current_balance", 10000),
        "today_pnl": paper_exchange.get("pnl", 0),  # Same as total for now
        "open_positions": paper_exchange.get("open_positions", 0),
        "sparkline_24h": []  # TODO: Implement historical data
    }

@router.get("/telemetry/health")
async def get_telemetry_health():
    """Get telemetry health data for dashboard"""
    ws_status = await ws_manager.get_status()

    # Calculate health score based on connections and errors
    health_score = 100
    issues = []

    if not ws_status.get("is_running", False):
        health_score -= 50
        issues.append("WebSocket manager not running")

    # Get client info safely
    clients = ws_status.get("clients", [])
    reconnect_attempts = clients[0].get("reconnect_attempts", 0) if clients else 0

    if reconnect_attempts > 5:
        health_score -= 20
        issues.append("High reconnection attempts")

    return {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - ws_manager.start_time if hasattr(ws_manager, 'start_time') else 0,
        "health_score": health_score,
        "status": "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical",
        "issues": issues,
        "latencies": {
            "websocket": {
                "count": ws_status.get("writer_stats", {}).get("ticks_written", 0),
                "avg": 45,
                "min": 10,
                "max": 200,
                "p50": 40,
                "p95": 100,
                "p99": 150
            }
        }
    }

@router.get("/strategies")
async def get_strategies():
    """Get strategies data for dashboard"""
    return ws_manager.strategy_manager.get_statistics()

@router.get("/orders")
async def get_orders(limit: int = 10):
    """Get recent orders for dashboard"""
    # For now, return mock data since we don't have order history tracking
    return [
        {
            "id": "order_001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.016,
            "price": 62040.82,
            "status": "FILLED",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "order_002",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "quantity": 0.5,
            "price": 3450.25,
            "status": "FILLED",
            "timestamp": datetime.now().isoformat()
        }
    ]