from fastapi import APIRouter
from app.core.monitoring.latency_tracker import LatencyTracker

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Singleton global
latency_tracker = LatencyTracker()

@router.get("/latency")
async def get_latency_stats():
    """Obtener estadísticas de latencia de todos los símbolos"""
    return latency_tracker.get_all_stats()

@router.get("/latency/{exchange}/{symbol}")
async def get_symbol_latency(exchange: str, symbol: str):
    """Obtener estadísticas de latencia de un símbolo específico"""
    stats = latency_tracker.get_stats(exchange, symbol)
    is_degraded = latency_tracker.check_degradation(exchange, symbol)

    return {
        **stats,
        "is_degraded": is_degraded
    }