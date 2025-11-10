from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class PositionSide(str, Enum):
    """Lado de la posición"""
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(str, Enum):
    """Estado de la posición"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

class Position(BaseModel):
    """
    Representa una posición abierta o cerrada
    """

    # Identificación
    id: Optional[str] = None
    strategy_name: str
    symbol: str
    side: PositionSide
    status: PositionStatus = PositionStatus.PENDING

    # Precios de entrada
    entry_price: float
    entry_time: datetime = Field(default_factory=utc_now)
    quantity: float

    # Gestión de riesgo
    stop_loss: float
    take_profit: float

    # Precios de salida
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None

    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Comisiones
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    # Metadata
    entry_signal: Optional[dict] = None
    exit_reason: Optional[str] = None

    class Config:
        use_enum_values = True

    def update_unrealized_pnl(self, current_price: float):
        """Actualizar PnL no realizado"""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

        # Restar comisiones
        self.unrealized_pnl -= (self.entry_commission + self.exit_commission)

    def close(self, exit_price: float, exit_reason: str, exit_commission: float = 0.0):
        """Cerrar posición"""
        self.exit_price = exit_price
        self.exit_time = datetime.now(timezone.utc)
        self.exit_reason = exit_reason
        self.exit_commission = exit_commission
        self.status = PositionStatus.CLOSED

        # Calcular PnL realizado
        if self.side == PositionSide.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity

        # Restar comisiones totales
        self.realized_pnl -= (self.entry_commission + self.exit_commission)

    def is_stop_hit(self, current_price: float) -> bool:
        """Verificar si se alcanzó el stop loss"""
        if self.side == PositionSide.LONG:
            return current_price <= self.stop_loss
        else:  # SHORT
            return current_price >= self.stop_loss

    def is_take_profit_hit(self, current_price: float) -> bool:
        """Verificar si se alcanzó el take profit"""
        if self.side == PositionSide.LONG:
            return current_price >= self.take_profit
        else:  # SHORT
            return current_price <= self.take_profit

    def get_duration_seconds(self) -> float:
        """Obtener duración de la posición en segundos"""
        end_time = self.exit_time if self.exit_time else datetime.now(timezone.utc)
        return (end_time - self.entry_time).total_seconds()

    def to_dict(self) -> dict:
        """Convertir a diccionario para logging"""
        return {
            "id": self.id,
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "duration_seconds": self.get_duration_seconds()
        }