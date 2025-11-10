# backend/app/core/execution/slippage_model.py

from typing import Dict, Optional
import random
from app.utils.logger import get_logger

logger = get_logger("slippage_model")

class SlippageModel:
    """
    Modelo de slippage realista basado en:
    - Volumen del orderbook
    - Volatilidad (ATR)
    - Tamaño de la orden
    - Spread actual
    """

    def __init__(
        self,
        base_slippage_pct: float = 0.05,  # 0.05% base
        volatility_multiplier: float = 0.5,
        volume_impact_factor: float = 0.3
    ):
        self.base_slippage_pct = base_slippage_pct
        self.volatility_multiplier = volatility_multiplier
        self.volume_impact_factor = volume_impact_factor

    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        orderbook: Optional[Dict] = None,
        atr: Optional[float] = None,
        spread_pct: Optional[float] = None
    ) -> float:
        """
        Calcular slippage en puntos de precio

        Returns:
            Slippage absoluto (siempre positivo)
        """

        # Slippage base
        base_slippage = price * (self.base_slippage_pct / 100)

        # Factor de volatilidad
        volatility_factor = 1.0
        if atr and price > 0:
            atr_pct = (atr / price) * 100
            volatility_factor = 1.0 + (atr_pct * self.volatility_multiplier)

        # Factor de volumen (impact del tamaño de orden)
        volume_factor = 1.0
        if orderbook:
            relevant_side = orderbook.get('asks' if side == 'BUY' else 'bids', [])
            if relevant_side:
                # Sumar volumen disponible en top 5 niveles
                available_volume = sum(qty for _, qty in relevant_side[:5])

                if available_volume > 0:
                    # Si la orden es grande vs volumen disponible, más slippage
                    order_impact = quantity / available_volume
                    volume_factor = 1.0 + (order_impact * self.volume_impact_factor)

        # Factor de spread
        spread_factor = 1.0
        if spread_pct:
            # Spread alto = más slippage
            spread_factor = 1.0 + (spread_pct * 2.0)

        # Slippage total
        total_slippage = base_slippage * volatility_factor * volume_factor * spread_factor

        # Añadir componente aleatorio (±20%)
        random_factor = random.uniform(0.8, 1.2)
        total_slippage *= random_factor

        logger.debug(
            "slippage_calculated",
            symbol=symbol,
            base=base_slippage,
            volatility_factor=volatility_factor,
            volume_factor=volume_factor,
            spread_factor=spread_factor,
            total=total_slippage
        )

        return total_slippage

    def apply_slippage(
        self,
        side: str,
        price: float,
        slippage: float
    ) -> float:
        """
        Aplicar slippage al precio

        BUY: precio sube (peor para comprador)
        SELL: precio baja (peor para vendedor)
        """
        if side == "BUY":
            return price + slippage
        else:  # SELL
            return price - slippage