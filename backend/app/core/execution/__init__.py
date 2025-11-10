# backend/app/core/execution/__init__.py

"""
Paper Exchange Module

Simulador de exchange para paper trading con comportamiento realista:
- Slippage basado en volatilidad y liquidez
- Comisiones maker/taker
- Fills parciales
- Latencia simulada
- Rechazos por condiciones de mercado
"""

from .order import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from .paper_exchange import PaperExchange
from .slippage_model import SlippageModel
from .commission_calculator import CommissionCalculator
from .fill_simulator import FillSimulator

__all__ = [
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "PaperExchange",
    "SlippageModel",
    "CommissionCalculator",
    "FillSimulator"
]