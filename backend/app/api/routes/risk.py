# backend/app/api/routes/risk.py

from fastapi import APIRouter, HTTPException, Body
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(prefix="/api/risk", tags=["risk"])

class KillSwitchRequest(BaseModel):
    """Request para activar kill-switch"""
    reason: str = "Manual emergency stop from dashboard"

@router.get("/status")
async def get_risk_status():
    """
    Obtener estado completo del Risk Manager

    Returns:
        - status: Estado de trading (enabled, kill-switch, cooldown)
        - balance: Balance actual y PnL
        - period_pnl: PnL diario y semanal
        - drawdown: Métricas de drawdown
        - position_limits: Límites y utilización de posiciones
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk Manager not initialized"
            )

        stats = ws_manager.risk_manager.get_statistics()

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "status": stats["status"],
                "balance": stats["balance"],
                "period_pnl": stats["period_pnl"],
                "drawdown": {
                    "current_pct": stats["drawdown"]["current"]["drawdown_pct"],
                    "current_amount": stats["drawdown"]["current"]["drawdown_amount"],
                    "in_drawdown": stats["drawdown"]["current"]["in_drawdown"],
                    "duration_seconds": stats["drawdown"]["current"]["duration_seconds"],
                    "max_historical_pct": stats["drawdown"]["historical"]["max_drawdown_pct"],
                    "max_historical_amount": stats["drawdown"]["historical"]["max_drawdown_amount"],
                    "limit_pct": stats["drawdown"]["limits"]["max_drawdown_pct"]
                },
                "position_limits": {
                    "open_positions": stats["position_limits"]["positions"]["open"],
                    "max_positions": stats["position_limits"]["positions"]["max"],
                    "utilization_pct": stats["position_limits"]["positions"]["utilization_pct"],
                    "total_exposure": stats["position_limits"]["exposure"]["total"],
                    "max_exposure": stats["position_limits"]["exposure"]["max"],
                    "exposure_utilization_pct": stats["position_limits"]["exposure"]["utilization_pct"]
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk status: {str(e)}"
        )

@router.get("/events")
async def get_risk_events(limit: int = 50):
    """
    Obtener eventos de riesgo recientes

    Args:
        limit: Número máximo de eventos a retornar (default: 50)

    Returns:
        Lista de eventos de riesgo con tipo, severidad, mensaje y timestamp
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk Manager not initialized"
            )

        # Obtener eventos recientes
        all_events = ws_manager.risk_manager.risk_events
        recent_events = all_events[-limit:] if len(all_events) > limit else all_events

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "events": [event.to_dict() for event in recent_events],
                "total_events": len(all_events)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk events: {str(e)}"
        )

@router.post("/kill-switch/activate")
async def activate_kill_switch(request: KillSwitchRequest):
    """
    Activar kill-switch manualmente

    Esto detendrá TODAS las operaciones de trading inmediatamente.
    Las posiciones abiertas permanecerán abiertas.
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk Manager not initialized"
            )

        # Activar kill-switch
        ws_manager.risk_manager.manual_kill_switch(reason=request.reason)

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Kill-switch activated successfully",
            "data": {
                "reason": request.reason,
                "kill_switch_active": True,
                "trading_enabled": False
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate kill-switch: {str(e)}"
        )

@router.post("/kill-switch/reset")
async def reset_kill_switch():
    """
    Desactivar kill-switch y reanudar trading

    PRECAUCIÓN: Solo usar después de revisar la causa del stop
    y confirmar que es seguro continuar trading.
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk Manager not initialized"
            )

        if not ws_manager.risk_manager.kill_switch_active:
            return {
                "success": False,
                "message": "Kill-switch is not active",
                "data": {
                    "kill_switch_active": False,
                    "trading_enabled": ws_manager.risk_manager.trading_enabled
                }
            }

        # Reset kill-switch
        ws_manager.risk_manager.reset_kill_switch()

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Kill-switch reset successfully - Trading resumed",
            "data": {
                "kill_switch_active": False,
                "trading_enabled": True
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset kill-switch: {str(e)}"
        )

@router.get("/limits")
async def get_risk_limits():
    """
    Obtener límites de riesgo configurados

    Returns:
        Todos los límites configurados del Risk Manager
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk Manager not initialized"
            )

        rm = ws_manager.risk_manager

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "loss_limits": {
                    "daily_loss_limit_pct": rm.max_daily_loss_pct,
                    "weekly_loss_limit_pct": rm.max_weekly_loss_pct,
                    "max_drawdown_pct": rm.drawdown_tracker.max_drawdown_pct
                },
                "trade_limits": {
                    "max_consecutive_losses": rm.max_consecutive_losses,
                    "cooldown_minutes": rm.cooldown_after_loss_minutes
                },
                "position_limits": {
                    "max_open_positions": rm.position_limits.max_open_positions,
                    "max_position_size_pct": rm.position_limits.max_position_size_pct,
                    "max_symbol_exposure_pct": rm.position_limits.max_symbol_exposure_pct,
                    "max_total_exposure_pct": rm.position_limits.max_total_exposure_pct
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk limits: {str(e)}"
        )

@router.get("/health")
async def get_risk_health():
    """
    Obtener health score del sistema de trading

    Returns:
        Score de 0-100 basado en múltiples factores de riesgo
    """
    try:
        from app.core.ws_manager import ws_manager

        if not ws_manager or not ws_manager.risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk Manager not initialized"
            )

        stats = ws_manager.risk_manager.get_statistics()

        # Calcular health score (0-100)
        health_score = 100
        health_factors = []

        # Factor 1: Kill-switch activo (-100)
        if stats["status"]["kill_switch_active"]:
            health_score = 0
            health_factors.append({
                "factor": "kill_switch",
                "impact": -100,
                "message": "Kill-switch is active"
            })
        else:
            # Factor 2: Drawdown (-0 a -40 puntos)
            drawdown_pct = stats["drawdown"]["current"]["drawdown_pct"]
            max_dd = stats["drawdown"]["limits"]["max_drawdown_pct"]
            drawdown_impact = -(drawdown_pct / max_dd) * 40
            health_score += drawdown_impact
            health_factors.append({
                "factor": "drawdown",
                "impact": drawdown_impact,
                "message": f"Current drawdown: {drawdown_pct:.2f}%"
            })

            # Factor 3: Pérdidas consecutivas (-0 a -20 puntos)
            consecutive = stats["status"]["consecutive_losses"]
            max_consecutive = 3
            consecutive_impact = -(consecutive / max_consecutive) * 20
            health_score += consecutive_impact
            health_factors.append({
                "factor": "consecutive_losses",
                "impact": consecutive_impact,
                "message": f"Consecutive losses: {consecutive}"
            })

            # Factor 4: PnL diario (-0 a -20 puntos)
            daily_pnl_pct = abs(stats["period_pnl"]["daily"]["pnl_pct"])
            daily_limit = stats["period_pnl"]["daily"]["limit_pct"]
            if stats["period_pnl"]["daily"]["pnl"] < 0:
                daily_impact = -(daily_pnl_pct / daily_limit) * 20
                health_score += daily_impact
                health_factors.append({
                    "factor": "daily_loss",
                    "impact": daily_impact,
                    "message": f"Daily loss: {daily_pnl_pct:.2f}%"
                })

            # Factor 5: Exposición de posiciones (-0 a -20 puntos)
            exposure_util = stats["position_limits"]["exposure"]["utilization_pct"]
            exposure_impact = -(exposure_util / 100) * 20
            health_score += exposure_impact
            health_factors.append({
                "factor": "position_exposure",
                "impact": exposure_impact,
                "message": f"Exposure utilization: {exposure_util:.1f}%"
            })

        # Asegurar que esté entre 0-100
        health_score = max(0, min(100, health_score))

        # Categoría de salud
        if health_score >= 80:
            health_category = "excellent"
            health_color = "green"
        elif health_score >= 60:
            health_category = "good"
            health_color = "blue"
        elif health_score >= 40:
            health_category = "moderate"
            health_color = "yellow"
        elif health_score >= 20:
            health_category = "poor"
            health_color = "orange"
        else:
            health_category = "critical"
            health_color = "red"

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "health_score": round(health_score, 1),
                "category": health_category,
                "color": health_color,
                "factors": health_factors,
                "recommendations": _get_health_recommendations(stats, health_score)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk health: {str(e)}"
        )

def _get_health_recommendations(stats: dict, health_score: float) -> list:
    """Generar recomendaciones basadas en health score"""
    recommendations = []

    if stats["status"]["kill_switch_active"]:
        recommendations.append({
            "priority": "critical",
            "message": "Kill-switch is active. Review the reason and reset when safe."
        })

    if stats["drawdown"]["current"]["in_drawdown"]:
        dd_pct = stats["drawdown"]["current"]["drawdown_pct"]
        if dd_pct > 7:
            recommendations.append({
                "priority": "high",
                "message": f"High drawdown ({dd_pct:.2f}%). Consider reducing position sizes."
            })

    if stats["status"]["consecutive_losses"] >= 2:
        recommendations.append({
            "priority": "medium",
            "message": "Multiple consecutive losses detected. Review strategy performance."
        })

    if stats["period_pnl"]["daily"]["pnl_pct"] < -1.5:
        recommendations.append({
            "priority": "medium",
            "message": "Approaching daily loss limit. Monitor closely."
        })

    if health_score >= 80:
        recommendations.append({
            "priority": "low",
            "message": "System health is excellent. Continue monitoring."
        })

    return recommendations