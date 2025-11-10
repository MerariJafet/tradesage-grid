from fastapi import APIRouter, HTTPException
from app.core.ws_manager import ws_manager
from typing import Optional

router = APIRouter(prefix="/strategies", tags=["strategies"])

@router.get("/")
async def get_all_strategies(symbol: Optional[str] = None):
    """Obtener todas las estrategias"""
    stats = ws_manager.strategy_manager.get_statistics(symbol)
    return stats

@router.get("/{symbol}/{strategy_name}")
async def get_strategy(symbol: str, strategy_name: str):
    """Obtener estrategia específica"""
    symbol = symbol.upper()

    strategy = ws_manager.strategy_manager.get_strategy(symbol, strategy_name)

    if not strategy:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy {strategy_name} not found for {symbol}"
        )

    return {
        "statistics": strategy.get_statistics(),
        "parameters": strategy.get_parameters() if hasattr(strategy, 'get_parameters') else {}
    }

@router.post("/{symbol}/{strategy_name}/enable")
async def enable_strategy(symbol: str, strategy_name: str):
    """Habilitar estrategia"""
    symbol = symbol.upper()

    ws_manager.strategy_manager.enable_strategy(symbol, strategy_name)

    return {"status": "enabled", "symbol": symbol, "strategy": strategy_name}

@router.post("/{symbol}/{strategy_name}/disable")
async def disable_strategy(symbol: str, strategy_name: str):
    """Deshabilitar estrategia"""
    symbol = symbol.upper()

    ws_manager.strategy_manager.disable_strategy(symbol, strategy_name)

    return {"status": "disabled", "symbol": symbol, "strategy": strategy_name}

@router.post("/{symbol}/add")
async def add_strategy(
    symbol: str,
    strategy_type: str,
    enabled: bool = True,
    **params
):
    """Añadir nueva estrategia"""
    symbol = symbol.upper()

    try:
        strategy = ws_manager.strategy_manager.add_strategy(
            symbol=symbol,
            strategy_type=strategy_type,
            enabled=enabled,
            **params
        )

        return {
            "status": "created",
            "symbol": symbol,
            "strategy_name": strategy.name,
            "strategy_type": strategy_type
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))