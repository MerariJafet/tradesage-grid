# backend/app/core/orders/limit_order.py

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from .base_order import BaseOrder, OrderSide, OrderStatus

class LimitOrder(BaseOrder):
    """Limit order - executes at specified price or better"""

    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 limit_price: float, timestamp: datetime = None, **kwargs):
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            timestamp=timestamp or datetime.now(timezone.utc),
            **kwargs
        )
        self.limit_price = limit_price

    def should_trigger(self, current_price: float, market_data: Dict[str, Any]) -> bool:
        """Check if limit order should trigger"""
        if self.side == OrderSide.BUY:
            # Buy limit: trigger when price <= limit_price
            return current_price <= self.limit_price
        else:  # SELL
            # Sell limit: trigger when price >= limit_price
            return current_price >= self.limit_price

    def get_trigger_price(self) -> Optional[float]:
        """Get the limit price"""
        return self.limit_price

    def __str__(self):
        return f"LimitOrder({self.symbol} {self.side.value.upper()} {self.quantity} @ {self.limit_price})"