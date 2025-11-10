from enum import Enum
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional

class RiskEventType(str, Enum):
    """Tipos de eventos de riesgo"""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_LIMIT = "position_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    EMERGENCY_STOP = "emergency_stop"
    WARNING = "warning"

class RiskEventSeverity(str, Enum):
    """Severidad del evento"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskEvent(BaseModel):
    """Evento de riesgo"""
    type: RiskEventType
    severity: RiskEventSeverity
    message: str
    value: float
    limit: float
    timestamp: datetime = datetime.now(timezone.utc)
    symbol: Optional[str] = None
    strategy: Optional[str] = None

    def should_stop_trading(self) -> bool:
        """Determinar si se debe detener el trading"""
        return self.severity in [RiskEventSeverity.CRITICAL, RiskEventSeverity.EMERGENCY]

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "limit": self.limit,
            "timestamp": str(self.timestamp),
            "symbol": self.symbol,
            "strategy": self.strategy
        }