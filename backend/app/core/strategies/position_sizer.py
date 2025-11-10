from typing import Optional
from app.config import settings
from app.utils.logger import get_logger
from app.core.position_sizing import (
    calculate_position_size as shared_calculate_position_size,
    DEFAULT_STARTING_BALANCE,
)

logger = get_logger("position_sizer")

class PositionSizer:
    """
    Calcula tamaño de posición basado en riesgo fijo

    Formula: Quantity = (Account * Risk%) / (ATR * Stop_Multiplier * Price)
    """

    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.risk_per_trade_pct = settings.MAX_RISK_PER_TRADE_PCT  # 0.5% default
        self.max_position_size_pct = 0.20  # Máximo 20% del capital por posición

    def calculate_quantity(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None
    ) -> float:
        """
        Calcular cantidad a operar basado en riesgo fijo

        Args:
            symbol: Símbolo del activo
            entry_price: Precio de entrada planeado
            stop_loss: Precio de stop loss
            atr: ATR actual (opcional, para validación)

        Returns:
            Cantidad a operar (en unidades del activo)
        """

        balance = self.account_balance if self.account_balance > 0 else DEFAULT_STARTING_BALANCE

        # shared helper already enforces max position size (10 % balance) and minimum size
        risk_fraction = self.risk_per_trade_pct / 100
        quantity = shared_calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_percent=risk_fraction,
            atr=atr,
        )

        # Redondear según el activo
        quantity = self._round_quantity(symbol, quantity)

        risk_amount = abs(entry_price - stop_loss) * quantity

        logger.info(
            "position_size_calculated",
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            quantity=quantity,
            risk_amount=risk_amount,
            notional=quantity * entry_price
        )

        return quantity

    # Compatibility wrapper for older strategies that call calculate_position_size
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None
    ) -> float:
        """
        Backwards-compatible alias for calculate_quantity.
        Older strategies may call calculate_position_size; forward to the
        canonical calculate_quantity implementation and log a short warning.
        """
        quantity = shared_calculate_position_size(
            balance=self.account_balance if self.account_balance > 0 else DEFAULT_STARTING_BALANCE,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_percent=self.risk_per_trade_pct / 100,
            atr=atr,
        )
        return self._round_quantity(symbol, quantity)

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Redondear cantidad según reglas del símbolo"""
        # TODO: Obtener step size del exchange
        # Por ahora, redondear a 3 decimales para crypto
        if "USDT" in symbol or "BUSD" in symbol:
            return round(quantity, 3)
        return round(quantity, 2)

    def validate_position_size(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Validar que el tamaño de posición cumple límites

        Returns:
            (is_valid, error_message)
        """
        notional = quantity * price

        # Verificar mínimo notional (ej: $10 en Binance)
        min_notional = 10.0
        if notional < min_notional:
            return False, f"Notional ${notional:.2f} below minimum ${min_notional}"

        # Verificar máximo por posición
        max_notional = self.account_balance * self.max_position_size_pct
        if notional > max_notional:
            return False, f"Notional ${notional:.2f} exceeds maximum ${max_notional:.2f}"

        return True, None

    def update_account_balance(self, new_balance: float):
        """Actualizar balance de cuenta"""
        self.account_balance = new_balance
        logger.info("position_sizer_balance_updated", new_balance=new_balance)