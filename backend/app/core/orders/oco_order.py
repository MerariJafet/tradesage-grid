# backend/app/core/orders/oco_order.py

from datetime import datetime
from typing import Optional, Dict, Any, List
from .base_order import BaseOrder, OrderSide, OrderStatus

class OCOOrder(BaseOrder):
    """One Cancels Other order - two orders where filling one cancels the other"""

    def __init__(self, symbol: str, quantity: float,
                 buy_price: Optional[float] = None,
                 sell_price: Optional[float] = None,
                 timestamp: datetime = None, **kwargs):
        # Determine side based on prices provided
        if buy_price is not None and sell_price is None:
            side = OrderSide.BUY
        elif sell_price is not None and buy_price is None:
            side = OrderSide.SELL
        else:
            raise ValueError("OCO order must specify either buy_price or sell_price, not both")

        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            timestamp=timestamp or datetime.utcnow(),
            **kwargs
        )

        self.buy_price = buy_price
        self.sell_price = sell_price
        self.other_order_cancelled = False

    def should_trigger(self, current_price: float, market_data: Dict[str, Any]) -> bool:
        """Check if OCO order should trigger"""
        if self.other_order_cancelled:
            return False  # Already cancelled

        if self.side == OrderSide.BUY and self.buy_price is not None:
            return current_price <= self.buy_price
        elif self.side == OrderSide.SELL and self.sell_price is not None:
            return current_price >= self.sell_price

        return False

    def get_trigger_price(self) -> Optional[float]:
        """Get the trigger price"""
        if self.side == OrderSide.BUY:
            return self.buy_price
        else:
            return self.sell_price

    def cancel_other(self):
        """Mark that the other order in the pair has been cancelled"""
        self.other_order_cancelled = True
        self.status = OrderStatus.CANCELLED

    def __str__(self):
        prices = []
        if self.buy_price:
            prices.append(f"buy@{self.buy_price}")
        if self.sell_price:
            prices.append(f"sell@{self.sell_price}")
        return f"OCOOrder({self.symbol} {self.quantity} {'/'.join(prices)})"