# backend/app/core/execution/commission_calculator.py

from typing import Dict, Literal
from app.utils.logger import get_logger

logger = get_logger("commission_calculator")

class CommissionCalculator:
    """
    Calculador de comisiones según tipo de orden y exchange

    Binance fees (ejemplo):
    - Maker: 0.02%
    - Taker: 0.04%
    """

    def __init__(self):
        # Estructura de comisiones por exchange y tier
        self.fee_structures = {
            "binance": {
                "spot": {
                    "maker": 0.02,  # 0.02%
                    "taker": 0.04   # 0.04%
                },
                "futures": {
                    "maker": 0.02,
                    "taker": 0.04
                }
            }
        }

    def calculate_commission(
        self,
        exchange: str,
        market_type: str,
        order_type: str,
        quantity: float,
        price: float,
        is_post_only: bool = False
    ) -> float:
        """
        Calcular comisión de la orden

        Args:
            exchange: Nombre del exchange
            market_type: 'spot' o 'futures'
            order_type: 'MARKET' o 'LIMIT'
            quantity: Cantidad operada
            price: Precio de ejecución
            is_post_only: Si la orden es post-only (siempre maker)

        Returns:
            Comisión en unidades monetarias (USDT)
        """

        # Obtener estructura de fees
        exchange_fees = self.fee_structures.get(exchange.lower(), self.fee_structures["binance"])
        market_fees = exchange_fees.get(market_type, exchange_fees["spot"])

        # Determinar si es maker o taker
        if is_post_only or order_type == "LIMIT":
            fee_pct = market_fees["maker"]
            fee_type = "maker"
        else:  # MARKET orders son siempre taker
            fee_pct = market_fees["taker"]
            fee_type = "taker"

        # Calcular comisión
        notional = quantity * price
        commission = notional * (fee_pct / 100)

        logger.debug(
            "commission_calculated",
            exchange=exchange,
            market_type=market_type,
            fee_type=fee_type,
            fee_pct=fee_pct,
            notional=notional,
            commission=commission
        )

        return commission