# backend/app/core/execution/fill_simulator.py

from typing import Optional, Dict, Tuple
import random
from app.core.execution.order import Order, OrderStatus
from app.utils.logger import get_logger

logger = get_logger("fill_simulator")

class FillSimulator:
    """
    Simula fills parciales y probabilidad de rechazo
    basado en condiciones de mercado
    """

    def __init__(
        self,
        base_fill_probability: float = 0.98,  # 98% de órdenes se llenan
        partial_fill_probability: float = 0.10  # 10% de fills parciales
    ):
        self.base_fill_probability = base_fill_probability
        self.partial_fill_probability = partial_fill_probability

    def simulate_fill(
        self,
        order: Order,
        orderbook: Optional[Dict] = None,
        spread_pct: Optional[float] = None
    ) -> Tuple[OrderStatus, float, Optional[str]]:
        """
        Simular ejecución de orden

        Returns:
            (status, filled_quantity, rejection_reason)
        """

        # Verificar rechazo
        rejection_reason = self._check_rejection(order, orderbook, spread_pct)
        if rejection_reason:
            return OrderStatus.REJECTED, 0.0, rejection_reason

        # Simular fill
        if order.time_in_force == "FOK":
            # Fill or Kill: todo o nada
            if random.random() < self.base_fill_probability:
                return OrderStatus.FILLED, order.quantity, None
            else:
                return OrderStatus.REJECTED, 0.0, "FOK: Unable to fill completely"

        elif order.time_in_force == "IOC":
            # Immediate or Cancel: puede ser parcial
            if random.random() < self.partial_fill_probability:
                fill_pct = random.uniform(0.5, 0.95)
                filled_qty = order.quantity * fill_pct
                return OrderStatus.PARTIALLY_FILLED, filled_qty, None
            else:
                return OrderStatus.FILLED, order.quantity, None

        else:  # GTC
            # Good Till Cancel: normalmente se llena completo
            if random.random() < self.base_fill_probability:
                return OrderStatus.FILLED, order.quantity, None
            else:
                # Muy raro que falle
                return OrderStatus.REJECTED, 0.0, "Insufficient liquidity"

    def _check_rejection(
        self,
        order: Order,
        orderbook: Optional[Dict],
        spread_pct: Optional[float]
    ) -> Optional[str]:
        """Verificar condiciones que causan rechazo"""

        # Rechazar si spread es muy amplio (> 0.5%)
        if spread_pct and spread_pct > 0.5:
            if random.random() < 0.3:  # 30% probabilidad de rechazo
                return f"Spread too wide: {spread_pct:.4f}%"

        # Rechazar si no hay suficiente liquidez
        if orderbook:
            relevant_side = orderbook.get('asks' if order.side == 'BUY' else 'bids', [])
            if relevant_side:
                available_volume = sum(qty for _, qty in relevant_side[:10])
                if order.quantity > available_volume * 2:
                    return "Insufficient liquidity for order size"

        return None