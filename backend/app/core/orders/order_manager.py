# backend/app/core/orders/order_manager.py

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .base_order import BaseOrder, OrderStatus
from app.utils.logger import get_logger

logger = get_logger("order_manager")

class OrderManager:
    """Manages multiple orders and their execution"""

    def __init__(self):
        self.orders: List[BaseOrder] = []
        self.filled_orders: List[BaseOrder] = []
        self.cancelled_orders: List[BaseOrder] = []

    def add_order(self, order: BaseOrder) -> str:
        """Add an order to the manager"""
        order_id = f"order_{len(self.orders) + 1}_{int(datetime.now(timezone.utc).timestamp())}"
        order.order_id = order_id
        self.orders.append(order)
        logger.info("order_added", order_id=order_id, type=type(order).__name__, symbol=order.symbol)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID"""
        for order in self.orders:
            if order.order_id == order_id and order.is_active():
                order.update_status(OrderStatus.CANCELLED)
                self.cancelled_orders.append(order)
                self.orders.remove(order)
                logger.info("order_cancelled", order_id=order_id)
                return True
        return False

    def get_active_orders(self, symbol: Optional[str] = None) -> List[BaseOrder]:
        """Get all active orders, optionally filtered by symbol"""
        active = [order for order in self.orders if order.is_active()]
        if symbol:
            active = [order for order in active if order.symbol == symbol]
        return active

    def process_market_data(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> List[BaseOrder]:
        """Process market data and trigger orders if conditions are met"""
        triggered_orders = []

        for order in self.orders[:]:  # Copy list to avoid modification during iteration
            if order.symbol == symbol and order.is_active():
                if order.should_trigger(current_price, market_data):
                    # In a real implementation, this would send to exchange
                    # For backtesting, we simulate fill at current price
                    order.fill(current_price, order.quantity - order.filled_quantity)
                    triggered_orders.append(order)
                    self.filled_orders.append(order)
                    self.orders.remove(order)

                    logger.info("order_triggered",
                              order_id=order.order_id,
                              symbol=symbol,
                              price=current_price,
                              quantity=order.filled_quantity)

        return triggered_orders

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific order"""
        for order in self.orders + self.filled_orders + self.cancelled_orders:
            if order.order_id == order_id:
                return {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "filled_quantity": order.filled_quantity,
                    "status": order.status.value,
                    "average_fill_price": order.average_fill_price,
                    "timestamp": order.timestamp.isoformat()
                }
        return None

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all orders and positions"""
        total_orders = len(self.orders) + len(self.filled_orders) + len(self.cancelled_orders)
        active_orders = len(self.get_active_orders())
        filled_orders = len(self.filled_orders)

        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": len(self.cancelled_orders)
        }

    def clear_all_orders(self):
        """Clear all orders (for testing/backtesting)"""
        self.orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()
        logger.info("all_orders_cleared")