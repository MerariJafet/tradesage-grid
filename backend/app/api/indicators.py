from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime
from app.core.indicators.indicator_manager import IndicatorManager
from app.utils.logger import get_logger

logger = get_logger("indicators_api")
router = APIRouter(prefix="/indicators", tags=["indicators"])

# Instancia global del manager
indicator_manager = IndicatorManager()

@router.get("/status")
async def get_indicators_status():
    """Estado general de los indicadores"""
    return {
        "active_symbols": list(indicator_manager.indicators.keys()),
        "total_symbols": len(indicator_manager.indicators),
        "cache_status": "connected" if indicator_manager.cache.redis else "disconnected"
    }

@router.get("/{symbol}")
async def get_symbol_indicators(
    symbol: str,
    indicators: Optional[List[str]] = Query(None, description="Lista de indicadores específicos")
):
    """
    Obtener valores de indicadores para un símbolo

    Args:
        symbol: Símbolo (ej: BTCUSDT)
        indicators: Lista opcional de indicadores específicos
    """
    try:
        if symbol not in indicator_manager.indicators:
            raise HTTPException(status_code=404, detail=f"Símbolo {symbol} no encontrado")

        all_values = indicator_manager.get_all_values(symbol)

        if not indicators:
            return all_values

        # Filtrar solo indicadores solicitados
        filtered = {}
        for ind in indicators:
            if ind in all_values:
                filtered[ind] = all_values[ind]
            else:
                logger.warning("indicator_not_found", symbol=symbol, indicator=ind)

        return filtered

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_indicators_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.get("/{symbol}/signals")
async def get_trading_signals(symbol: str):
    """
    Obtener señales de trading para un símbolo

    Returns:
        Dict con señales booleanas y valores numéricos
    """
    try:
        if symbol not in indicator_manager.indicators:
            raise HTTPException(status_code=404, detail=f"Símbolo {symbol} no encontrado")

        signals = indicator_manager.get_signals(symbol)

        return {
            "symbol": symbol,
            "signals": signals,
            "timestamp": str(datetime.utcnow())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_signals_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.get("/{symbol}/ready")
async def check_indicators_ready(
    symbol: str,
    required_indicators: Optional[List[str]] = Query(None)
):
    """
    Verificar si indicadores están listos para operar

    Args:
        symbol: Símbolo
        required_indicators: Lista opcional de indicadores requeridos
    """
    try:
        ready = indicator_manager.is_ready(symbol, required_indicators)

        return {
            "symbol": symbol,
            "ready": ready,
            "required_indicators": required_indicators or "all"
        }

    except Exception as e:
        logger.error("check_ready_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.post("/{symbol}/reset")
async def reset_symbol_indicators(symbol: str):
    """Resetear indicadores de un símbolo"""
    try:
        indicator_manager.reset_symbol(symbol)

        return {
            "symbol": symbol,
            "status": "reset",
            "timestamp": str(datetime.utcnow())
        }

    except Exception as e:
        logger.error("reset_indicators_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.get("/{symbol}/cache")
async def get_cached_indicators(symbol: str):
    """
    Obtener indicadores desde caché Redis

    Returns:
        Dict con valores cacheados
    """
    try:
        cached = indicator_manager.cache.get_all_symbol(symbol)

        return {
            "symbol": symbol,
            "cached_indicators": cached,
            "cache_timestamp": str(datetime.utcnow())
        }

    except Exception as e:
        logger.error("get_cache_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.get("/{symbol}/detailed")
async def get_detailed_indicators(symbol: str):
    """
    Obtener información detallada de indicadores

    Incluye estado interno, buffers, etc.
    """
    try:
        if symbol not in indicator_manager.indicators:
            raise HTTPException(status_code=404, detail=f"Símbolo {symbol} no encontrado")

        detailed = {}

        for name, indicator in indicator_manager.indicators[symbol].items():
            detailed[name] = {
                "ready": getattr(indicator, 'is_ready', True),
                "last_value": getattr(indicator, 'last_value', None),
                "state": indicator.get_state()
            }

        return {
            "symbol": symbol,
            "indicators": detailed,
            "timestamp": str(datetime.utcnow())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_detailed_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")