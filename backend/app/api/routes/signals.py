from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List, Optional

router = APIRouter(prefix="/api/signals", tags=["signals"])

@router.get("/aggregated")
async def get_aggregated_signals(limit: int = 50):
    """
    Obtener señales agregadas recientes

    Returns:
        Lista de señales agregadas con estrategias fuente
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.strategy_manager:
            raise HTTPException(
                status_code=503,
                detail="Strategy Manager not initialized"
            )

        # Obtener señales del aggregator
        aggregator = ws_manager.strategy_manager.signal_aggregator
        recent_signals = aggregator.get_recent_aggregated_signals(limit)

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "signals": recent_signals,
                "total": len(recent_signals)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get aggregated signals: {str(e)}"
        )

@router.get("/by-strategy/{strategy_name}")
async def get_signals_by_strategy(strategy_name: str, limit: int = 20):
    """
    Obtener señales de una estrategia específica

    Args:
        strategy_name: "BreakoutCompression" o "MeanReversion"
        limit: Número máximo de señales
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.strategy_manager:
            raise HTTPException(
                status_code=503,
                detail="Strategy Manager not initialized"
            )

        # Recolectar señales de todas las estrategias del tipo especificado
        signals = []

        for symbol, symbol_strategies in ws_manager.strategy_manager.strategies.items():
            for name, strategy in symbol_strategies.items():
                if name == strategy_name:
                    # Obtener estadísticas de la estrategia
                    stats = strategy.get_statistics()
                    signals.append({
                        "symbol": symbol,
                        "strategy": name,
                        "enabled": strategy.enabled,
                        "total_signals_generated": stats.get('total_signals_generated', 0),
                        "total_signals_executed": stats.get('total_signals_executed', 0),
                        "win_rate": stats.get('win_rate', 0.0),
                        "total_pnl": stats.get('total_pnl', 0.0)
                    })

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "strategy_name": strategy_name,
                "signals": signals[:limit]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get signals: {str(e)}"
        )

@router.get("/summary")
async def get_signals_summary():
    """
    Obtener resumen general de todas las señales y estrategias
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.strategy_manager:
            raise HTTPException(
                status_code=503,
                detail="Strategy Manager not initialized"
            )

        # Obtener estadísticas del strategy manager
        stats = ws_manager.strategy_manager.get_statistics()

        # Obtener resumen del aggregator
        aggregator = ws_manager.strategy_manager.signal_aggregator
        summary = {
            "total_strategies": stats.get('global', {}).get('total_strategies', 0),
            "enabled_strategies": stats.get('global', {}).get('enabled_strategies', 0),
            "total_open_positions": stats.get('global', {}).get('total_open_positions', 0),
            "account_balance": stats.get('global', {}).get('account_balance', 0.0),
            "recent_aggregated_signals": len(aggregator.get_recent_aggregated_signals(10))
        }

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get signals summary: {str(e)}"
        )