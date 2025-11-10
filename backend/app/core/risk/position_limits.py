from typing import Dict, Optional
from app.utils.logger import get_logger

logger = get_logger("position_limits")

class PositionLimits:
    """
    Gestor de límites de posición

    Límites:
    - Max posiciones abiertas simultáneas
    - Max exposición por símbolo
    - Max exposición total
    - Max tamaño de posición individual
    """

    def __init__(
        self,
        account_balance: float,
        max_open_positions: int = 3,
        max_position_size_pct: float = 2.0,  # 2% del balance por posición
        max_symbol_exposure_pct: float = 5.0,  # 5% del balance por símbolo
        max_total_exposure_pct: float = 10.0  # 10% del balance total en riesgo
    ):
        self.account_balance = account_balance
        self.max_open_positions = max_open_positions
        self.max_position_size_pct = max_position_size_pct
        self.max_symbol_exposure_pct = max_symbol_exposure_pct
        self.max_total_exposure_pct = max_total_exposure_pct

        # Tracking
        self.open_positions: Dict[str, float] = {}  # position_id -> risk_amount
        self.symbol_exposure: Dict[str, float] = {}  # symbol -> total_exposure

    def can_open_position(
        self,
        symbol: str,
        position_size: float,
        risk_amount: float
    ) -> tuple[bool, Optional[str]]:
        """
        Verificar si se puede abrir una posición

        Returns:
            (can_open, reason_if_not)
        """

        # 1. Verificar número de posiciones
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions reached ({self.max_open_positions})"

        # 2. Verificar tamaño de posición individual
        max_position_size = self.account_balance * (self.max_position_size_pct / 100)
        if risk_amount > max_position_size:
            return False, f"Position risk ${risk_amount:.2f} exceeds limit ${max_position_size:.2f}"

        # 3. Verificar exposición por símbolo
        current_symbol_exposure = self.symbol_exposure.get(symbol, 0.0)
        new_symbol_exposure = current_symbol_exposure + risk_amount
        max_symbol_exposure = self.account_balance * (self.max_symbol_exposure_pct / 100)

        if new_symbol_exposure > max_symbol_exposure:
            return False, f"Symbol exposure ${new_symbol_exposure:.2f} would exceed limit ${max_symbol_exposure:.2f}"

        # 4. Verificar exposición total
        current_total_exposure = sum(self.open_positions.values())
        new_total_exposure = current_total_exposure + risk_amount
        max_total_exposure = self.account_balance * (self.max_total_exposure_pct / 100)

        if new_total_exposure > max_total_exposure:
            return False, f"Total exposure ${new_total_exposure:.2f} would exceed limit ${max_total_exposure:.2f}"

        return True, None

    def register_position_opened(
        self,
        position_id: str,
        symbol: str,
        risk_amount: float
    ):
        """Registrar apertura de posición"""
        self.open_positions[position_id] = risk_amount
        self.symbol_exposure[symbol] = self.symbol_exposure.get(symbol, 0.0) + risk_amount

        logger.info(
            "position_opened",
            position_id=position_id,
            symbol=symbol,
            risk_amount=risk_amount,
            open_positions=len(self.open_positions),
            total_exposure=sum(self.open_positions.values())
        )

    def register_position_closed(
        self,
        position_id: str,
        symbol: str,
        risk_amount: float
    ):
        """Registrar cierre de posición"""
        if position_id in self.open_positions:
            del self.open_positions[position_id]

        self.symbol_exposure[symbol] = max(0, self.symbol_exposure.get(symbol, 0.0) - risk_amount)

        logger.info(
            "position_closed",
            position_id=position_id,
            symbol=symbol,
            open_positions=len(self.open_positions),
            total_exposure=sum(self.open_positions.values())
        )

    def update_balance(self, new_balance: float):
        """Actualizar balance de cuenta"""
        self.account_balance = new_balance

    def get_statistics(self) -> dict:
        """Obtener estadísticas de límites"""
        total_exposure = sum(self.open_positions.values())
        max_total = self.account_balance * (self.max_total_exposure_pct / 100)

        return {
            "positions": {
                "open": len(self.open_positions),
                "max": self.max_open_positions,
                "utilization_pct": (len(self.open_positions) / self.max_open_positions) * 100
            },
            "exposure": {
                "total": total_exposure,
                "max": max_total,
                "utilization_pct": (total_exposure / max_total * 100) if max_total > 0 else 0,
                "by_symbol": self.symbol_exposure.copy()
            },
            "limits": {
                "max_open_positions": self.max_open_positions,
                "max_position_size_pct": self.max_position_size_pct,
                "max_symbol_exposure_pct": self.max_symbol_exposure_pct,
                "max_total_exposure_pct": self.max_total_exposure_pct
            }
        }