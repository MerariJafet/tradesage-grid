# backend/app/core/orders/iceberg_order.py

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from .base_order import BaseOrder, OrderSide, OrderStatus

class IcebergOrder(BaseOrder):
    """Iceberg order - displays only a portion of total quantity to hide full size"""

    def __init__(self, symbol: str, side: OrderSide, total_quantity: float,
                 display_quantity: float, limit_price: float,
                 timestamp: datetime = None, **kwargs):
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=total_quantity,  # Total quantity
            timestamp=timestamp or datetime.now(timezone.utc),
            **kwargs
        )
        self.display_quantity = display_quantity  # Visible quantity
        self.limit_price = limit_price
        self.remaining_hidden_quantity = total_quantity

    def should_trigger(self, current_price: float, market_data: Dict[str, Any]) -> bool:
        """Check if iceberg order should trigger"""
        if self.side == OrderSide.BUY:
            return current_price <= self.limit_price
        else:  # SELL
            return current_price >= self.limit_price

    def get_trigger_price(self) -> Optional[float]:
        """Get the limit price"""
        return self.limit_price

    def get_display_quantity(self) -> float:
        """Get the currently displayed quantity"""
        return min(self.display_quantity, self.remaining_hidden_quantity)

    def fill_displayed(self, price: float, quantity: float):
        """Fill the displayed portion and refresh if more hidden quantity exists"""
        actual_fill = min(quantity, self.remaining_hidden_quantity)
        self.fill(price, actual_fill)
        self.remaining_hidden_quantity -= actual_fill

    def is_fully_hidden(self) -> bool:
        """Check if there's still hidden quantity"""
        return self.remaining_hidden_quantity > 0

    def __str__(self):
        return f"IcebergOrder({self.symbol} {self.side.value.upper()} {self.quantity} display:{self.display_quantity} @ {self.limit_price})"