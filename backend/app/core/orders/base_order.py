# backend/app/core/orders/base_order.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class BaseOrder(ABC):
    """Base class for all order types"""

    symbol: str
    side: OrderSide
    quantity: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None

    @abstractmethod
    def should_trigger(self, current_price: float, market_data: Dict[str, Any]) -> bool:
        """Check if order should be triggered based on current market conditions"""
        pass

    @abstractmethod
    def get_trigger_price(self) -> Optional[float]:
        """Get the price at which this order should trigger"""
        pass

    def update_status(self, new_status: OrderStatus):
        """Update order status"""
        self.status = new_status

    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]

    def fill(self, price: float, quantity: float):
        """Fill part or all of the order"""
        if quantity > self.quantity - self.filled_quantity:
            quantity = self.quantity - self.filled_quantity

        if self.filled_quantity == 0:
            self.average_fill_price = price
        else:
            total_value = (self.filled_quantity * self.average_fill_price) + (quantity * price)
            self.average_fill_price = total_value / (self.filled_quantity + quantity)

        self.filled_quantity += quantity

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL