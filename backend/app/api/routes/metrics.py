from fastapi import APIRouter
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter()

# Métricas custom
trades_executed = Counter(
    'tradesage_trades_executed_total',
    'Total trades executed',
    ['exchange', 'symbol', 'strategy', 'mode']
)

trade_pnl = Histogram(
    'tradesage_trade_pnl',
    'PnL per trade',
    ['strategy', 'mode']
)

ws_latency = Histogram(
    'tradesage_ws_latency_seconds',
    'WebSocket message latency',
    ['exchange', 'symbol']
)

active_positions = Gauge(
    'tradesage_active_positions',
    'Number of active positions',
    ['exchange', 'symbol']
)

@router.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )