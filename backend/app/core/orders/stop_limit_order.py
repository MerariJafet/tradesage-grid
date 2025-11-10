# backend/app/core/orders/stop_limit_order.py

from datetime import datetime
from typing import Optional, Dict, Any
from .base_order import BaseOrder, OrderSide, OrderStatus

class StopLimitOrder(BaseOrder):
    """Stop-limit order - combines stop price with limit price"""

    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 stop_price: float, limit_price: float, timestamp: datetime = None, **kwargs):
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            timestamp=timestamp or datetime.utcnow(),
            **kwargs
        )
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.stop_triggered = False

    def should_trigger(self, current_price: float, market_data: Dict[str, Any]) -> bool:
        """Check if stop-limit order should trigger"""
        if self.stop_triggered:
            # Once stop is triggered, behave like a limit order
            if self.side == OrderSide.BUY:
                return current_price <= self.limit_price
            else:  # SELL
                return current_price >= self.limit_price
        else:
            # Check if stop price is hit
            if self.side == OrderSide.BUY:
                # Buy stop: trigger when price >= stop_price
                if current_price >= self.stop_price:
                    self.stop_triggered = True
                    return True
            else:  # SELL
                # Sell stop: trigger when price <= stop_price
                if current_price <= self.stop_price:
                    self.stop_triggered = True
                    return True
            return False

    def get_trigger_price(self) -> Optional[float]:
        """Get the current trigger price"""
        if self.stop_triggered:
            return self.limit_price
        else:
            return self.stop_price

    def __str__(self):
        return f"StopLimitOrder({self.symbol} {self.side.value.upper()} {self.quantity} stop@{self.stop_price} limit@{self.limit_price})"