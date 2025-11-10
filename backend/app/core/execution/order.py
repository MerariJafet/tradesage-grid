# backend/app/core/execution/order.py

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
from enum import Enum
import uuid


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class OrderSide(str, Enum):
    """Lado de la orden"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    """Tipo de orden"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderStatus(str, Enum):
    """Estado de la orden"""
    PENDING = "PENDING"          # Creada, esperando procesamiento
    SUBMITTED = "SUBMITTED"      # Enviada al exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"            # Completamente ejecutada
    REJECTED = "REJECTED"        # Rechazada
    CANCELLED = "CANCELLED"      # Cancelada por el usuario
    EXPIRED = "EXPIRED"          # Expiró (para órdenes con TTL)

class TimeInForce(str, Enum):
    """Time in Force"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

class Order(BaseModel):
    """Modelo de orden de trading"""

    # Identificación
    id: str = Field(default_factory=lambda: f"order_{uuid.uuid4().hex[:12]}")
    strategy_name: str
    symbol: str

    # Tipo y lado
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.GTC

    # Precios y cantidad
    quantity: float
    price: Optional[float] = None  # None para market orders
    stop_price: Optional[float] = None  # Para stop loss

    # Estado
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None

    # Costos
    commission: float = 0.0
    slippage: float = 0.0

    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Metadata
    rejection_reason: Optional[str] = None
    mode: Literal["paper", "live"] = "paper"

    class Config:
        use_enum_values = True

    def is_fully_filled(self) -> bool:
        """Verificar si la orden está completamente ejecutada"""
        return abs(self.filled_quantity - self.quantity) < 1e-8

    def get_fill_percentage(self) -> float:
        """Obtener porcentaje de fill"""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    def get_total_cost(self) -> float:
        """Obtener costo total (comisión + slippage)"""
        return self.commission + abs(self.slippage)

    def to_dict(self) -> dict:
        """Convertir a diccionario para logging"""
        return {
            "id": self.id,
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "status": self.status,
            "commission": self.commission,
            "slippage": self.slippage,
            "created_at": str(self.created_at)
        }