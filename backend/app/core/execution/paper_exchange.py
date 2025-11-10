# backend/app/core/execution/paper_exchange.py

from typing import Dict, Optional, List
import asyncio
from datetime import datetime, timezone
import random
from app.core.execution.order import Order, OrderStatus, OrderSide
from app.core.execution.slippage_model import SlippageModel
from app.core.execution.commission_calculator import CommissionCalculator
from app.core.execution.fill_simulator import FillSimulator
from app.core.telemetry.metrics import telemetry_system
from app.utils.logger import get_logger

logger = get_logger("paper_exchange")

class PaperExchange:
    """
    Simulador de exchange para paper trading

    Simula:
    - Ejecución de órdenes
    - Slippage realista
    - Comisiones maker/taker
    - Fills parciales
    - Latencia
    - Rechazos
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        exchange_name: str = "binance",
        market_type: str = "futures"
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.exchange_name = exchange_name
        self.market_type = market_type

        # Componentes
        self.slippage_model = SlippageModel()
        self.commission_calculator = CommissionCalculator()
        self.fill_simulator = FillSimulator()

        # Estado
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trades_executed = 0
        self.total_commission_paid = 0.0

        logger.info(
            "paper_exchange_initialized",
            initial_balance=initial_balance,
            exchange=exchange_name,
            market_type=market_type
        )

    async def submit_order(
        self,
        order: Order,
        current_price: float,
        orderbook: Optional[Dict] = None,
        atr: Optional[float] = None,
        spread_pct: Optional[float] = None
    ) -> Order:
        """
        Enviar orden al paper exchange

        Args:
            order: Orden a ejecutar
            current_price: Precio actual del mercado
            orderbook: Order book actual (opcional)
            atr: ATR actual (opcional, para slippage)
            spread_pct: Spread actual en % (opcional)

        Returns:
            Orden actualizada con resultado de ejecución
        """

        logger.info(
            "order_submitted",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            type=order.order_type,
            quantity=order.quantity,
            price=order.price or current_price
        )

        order.submitted_at = datetime.now(timezone.utc)
        order.status = OrderStatus.SUBMITTED

        # Simular latencia de ejecución (10-50ms)
        await asyncio.sleep(random.uniform(0.01, 0.05))

        # Determinar precio de ejecución
        execution_price = order.price if order.price else current_price

        # Calcular slippage
        slippage = self.slippage_model.calculate_slippage(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            orderbook=orderbook,
            atr=atr,
            spread_pct=spread_pct
        )

        # Aplicar slippage al precio
        filled_price = self.slippage_model.apply_slippage(
            side=order.side,
            price=execution_price,
            slippage=slippage
        )

        # Simular fill
        status, filled_quantity, rejection_reason = self.fill_simulator.simulate_fill(
            order=order,
            orderbook=orderbook,
            spread_pct=spread_pct
        )

        # Actualizar orden
        order.status = status
        order.filled_quantity = filled_quantity
        order.filled_price = filled_price
        order.slippage = slippage
        order.rejection_reason = rejection_reason

        if status == OrderStatus.REJECTED:
            logger.warning(
                "order_rejected",
                order_id=order.id,
                reason=rejection_reason
            )
            # ✨ TELEMETRY: Registrar orden rechazada
            exec_sensor = telemetry_system.get_execution_sensor(order.symbol)
            exec_sensor.record_order(
                expected_price=execution_price,
                filled_price=filled_price,
                expected_quantity=order.quantity,
                filled_quantity=0.0,
                rejected=True,
                rejection_reason=rejection_reason
            )
            self.orders[order.id] = order
            return order

        # Calcular comisión
        commission = self.commission_calculator.calculate_commission(
            exchange=self.exchange_name,
            market_type=self.market_type,
            order_type=order.order_type,
            quantity=filled_quantity,
            price=filled_price,
            is_post_only=False
        )

        order.commission = commission
        order.filled_at = datetime.now(timezone.utc)

        # Actualizar balance y posiciones
        self._update_balance_and_positions(order)

        # ✨ TELEMETRY: Registrar orden ejecutada exitosamente
        exec_sensor = telemetry_system.get_execution_sensor(order.symbol)
        exec_sensor.record_order(
            expected_price=execution_price,
            filled_price=filled_price,
            expected_quantity=order.quantity,
            filled_quantity=filled_quantity,
            rejected=False
        )

        # Guardar orden
        self.orders[order.id] = order
        self.trades_executed += 1
        self.total_commission_paid += commission

        logger.info(
            "order_filled",
            order_id=order.id,
            status=status,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            slippage=slippage,
            commission=commission,
            balance=self.current_balance
        )

        return order

    def _update_balance_and_positions(self, order: Order):
        """Actualizar balance y posiciones después de fill"""

        notional = order.filled_quantity * order.filled_price

        if order.side == OrderSide.BUY:
            # Compra: reduce balance, aumenta posición
            self.current_balance -= (notional + order.commission)
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.filled_quantity
        else:  # SELL
            # Venta: aumenta balance, reduce posición
            self.current_balance += (notional - order.commission)
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - order.filled_quantity

    def get_position(self, symbol: str) -> float:
        """Obtener posición actual de un símbolo"""
        return self.positions.get(symbol, 0.0)

    def get_pnl(self) -> float:
        """Obtener PnL total"""
        return self.current_balance - self.initial_balance

    def get_pnl_percent(self) -> float:
        """Obtener PnL en porcentaje"""
        if self.initial_balance == 0:
            return 0.0
        return (self.get_pnl() / self.initial_balance) * 100

    def get_statistics(self) -> Dict:
        """Obtener estadísticas del paper exchange"""
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "pnl": self.get_pnl(),
            "pnl_percent": self.get_pnl_percent(),
            "trades_executed": self.trades_executed,
            "total_commission_paid": self.total_commission_paid,
            "open_positions": len([p for p in self.positions.values() if abs(p) > 1e-8]),
            "positions": self.positions.copy()
        }

    def get_order(self, order_id: str) -> Optional[Order]:
        """Obtener orden por ID"""
        return self.orders.get(order_id)

    def get_recent_orders(self, limit: int = 10) -> List[Order]:
        """Obtener órdenes recientes"""
        sorted_orders = sorted(
            self.orders.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return sorted_orders[:limit]