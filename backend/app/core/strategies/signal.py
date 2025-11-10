from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class SignalAction(str, Enum):
    """Acciones posibles de señal"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class SignalType(str, Enum):
    """Tipo de señal según estrategia"""
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    AGGREGATED = "aggregated"

class TradingSignal(BaseModel):
    """
    Señal de trading generada por una estrategia
    """

    # Identificación
    strategy_name: str
    symbol: str
    signal_type: SignalType
    action: SignalAction

    # Precios y sizing
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float

    # Confianza y validación
    confidence: float = Field(ge=0, le=1, description="Confianza de la señal (0-1)")

    # Contexto
    indicators: dict = Field(default_factory=dict, description="Indicadores al momento de la señal")
    reason: str = Field(default="", description="Razón de la señal")
    source_signals: list[str] = Field(default_factory=list, description="Señales fuente para señales agregadas")

    # Metadata
    timestamp: datetime = Field(default_factory=utc_now)
    expiry_seconds: Optional[int] = Field(default=None, description="Tiempo de expiración de la señal")

    # Validación
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "strategy_name": "BreakoutCompression",
                "symbol": "BTCUSDT",
                "signal_type": "breakout",
                "action": "BUY",
                "entry_price": 62100.0,
                "stop_loss": 62000.0,
                "take_profit": 62250.0,
                "quantity": 0.016,
                "confidence": 0.75,
                "indicators": {
                    "bb_bandwidth": 0.015,
                    "atr": 125.50,
                    "rsi_2": 45.0
                },
                "reason": "BB compression + breakout above upper band"
            }
        }

    def get_risk_amount(self) -> float:
        """Calcular monto en riesgo"""
        return abs(self.entry_price - self.stop_loss) * self.quantity

    def get_risk_reward_ratio(self) -> float:
        """Calcular ratio riesgo/recompensa"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0

    def is_expired(self) -> bool:
        """Verificar si la señal ha expirado"""
        if self.expiry_seconds is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return elapsed > self.expiry_seconds