# backend/app/core/orders/trailing_stop_order.py

from datetime import datetime
from typing import Optional, Dict, Any
from .base_order import BaseOrder, OrderSide, OrderStatus

class TrailingStopOrder(BaseOrder):
    """Trailing stop order - follows price movement with dynamic stop level"""

    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 trailing_percent: float, initial_stop_price: Optional[float] = None,
                 timestamp: datetime = None, **kwargs):
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            timestamp=timestamp or datetime.utcnow(),
            **kwargs
        )
        self.trailing_percent = trailing_percent / 100.0  # Convert to decimal
        self.stop_price = initial_stop_price
        self.highest_price = initial_stop_price if initial_stop_price else 0
        self.lowest_price = initial_stop_price if initial_stop_price else float('inf')

    def should_trigger(self, current_price: float, market_data: Dict[str, Any]) -> bool:
        """Check if trailing stop should trigger"""
        self._update_trailing_stop(current_price)

        if self.side == OrderSide.SELL:
            # Sell trailing stop: trigger when price falls to stop level
            return current_price <= self.stop_price
        else:  # BUY
            # Buy trailing stop: trigger when price rises to stop level
            return current_price >= self.stop_price

    def get_trigger_price(self) -> Optional[float]:
        """Get the current stop price"""
        return self.stop_price

    def _update_trailing_stop(self, current_price: float):
        """Update the trailing stop based on price movement"""
        if self.side == OrderSide.SELL:
            # For sell orders, trail below the highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.stop_price = current_price * (1 - self.trailing_percent)
        else:  # BUY
            # For buy orders, trail above the lowest price
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                self.stop_price = current_price * (1 + self.trailing_percent)

    def set_initial_price(self, entry_price: float):
        """Set initial price for trailing calculation"""
        if self.side == OrderSide.SELL:
            self.highest_price = entry_price
            self.stop_price = entry_price * (1 - self.trailing_percent)
        else:  # BUY
            self.lowest_price = entry_price
            self.stop_price = entry_price * (1 + self.trailing_percent)

    def __str__(self):
        return f"TrailingStopOrder({self.symbol} {self.side.value.upper()} {self.quantity} trail:{self.trailing_percent*100:.1f}% stop@{self.stop_price:.2f})"