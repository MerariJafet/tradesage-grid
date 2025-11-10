from typing import List, Optional
from datetime import datetime, timedelta
from app.utils.logger import get_logger

logger = get_logger("drawdown_tracker")

class DrawdownTracker:
    """
    Tracker de drawdowns (caídas desde peak)

    Métricas:
    - Current drawdown
    - Max drawdown
    - Drawdown duration
    - Recovery time
    """

    def __init__(
        self,
        initial_balance: float,
        max_drawdown_pct: float = 10.0  # 10% max drawdown
    ):
        self.initial_balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct

        # Estado
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        self.trough_balance = initial_balance

        # Drawdown actual
        self.current_drawdown_pct = 0.0
        self.current_drawdown_amount = 0.0

        # Máximo drawdown histórico
        self.max_drawdown_pct_historical = 0.0
        self.max_drawdown_amount_historical = 0.0

        # Timestamps
        self.peak_timestamp = datetime.utcnow()
        self.trough_timestamp = datetime.utcnow()
        self.drawdown_start = None
        self.last_update = datetime.utcnow()

        # Historia
        self.drawdown_events: List[dict] = []

    def update(self, current_balance: float) -> Optional[dict]:
        """
        Actualizar tracker con nuevo balance

        Returns:
            dict con info de drawdown si hay cambio significativo
        """
        self.current_balance = current_balance
        self.last_update = datetime.utcnow()

        # Nuevo peak
        if current_balance > self.peak_balance:
            # Si estábamos en drawdown, registrar recuperación
            if self.current_drawdown_pct > 0:
                self._record_recovery()

            self.peak_balance = current_balance
            self.peak_timestamp = datetime.utcnow()
            self.current_drawdown_pct = 0.0
            self.current_drawdown_amount = 0.0
            self.drawdown_start = None

            logger.info(
                "new_peak",
                balance=current_balance,
                previous_peak=self.peak_balance
            )

            return None

        # Calcular drawdown actual
        drawdown_amount = self.peak_balance - current_balance
        drawdown_pct = (drawdown_amount / self.peak_balance) * 100

        # Actualizar drawdown actual
        self.current_drawdown_amount = drawdown_amount
        self.current_drawdown_pct = drawdown_pct

        # Iniciar tracking de drawdown
        if self.drawdown_start is None and drawdown_pct > 0.1:  # > 0.1%
            self.drawdown_start = datetime.utcnow()

        # Nuevo trough (punto más bajo)
        if current_balance < self.trough_balance:
            self.trough_balance = current_balance
            self.trough_timestamp = datetime.utcnow()

        # Actualizar máximo histórico
        if drawdown_pct > self.max_drawdown_pct_historical:
            self.max_drawdown_pct_historical = drawdown_pct
            self.max_drawdown_amount_historical = drawdown_amount

            logger.warning(
                "new_max_drawdown",
                drawdown_pct=drawdown_pct,
                drawdown_amount=drawdown_amount,
                from_balance=self.peak_balance,
                to_balance=current_balance
            )

        # Verificar si excede límite
        if drawdown_pct >= self.max_drawdown_pct:
            return {
                "type": "max_drawdown_exceeded",
                "current_drawdown_pct": drawdown_pct,
                "limit": self.max_drawdown_pct,
                "amount": drawdown_amount,
                "duration_seconds": self._get_drawdown_duration()
            }

        # Warning si está cerca del límite (80%)
        elif drawdown_pct >= (self.max_drawdown_pct * 0.8):
            return {
                "type": "drawdown_warning",
                "current_drawdown_pct": drawdown_pct,
                "limit": self.max_drawdown_pct,
                "amount": drawdown_amount,
                "duration_seconds": self._get_drawdown_duration()
            }

        return None

    def _get_drawdown_duration(self) -> int:
        """Obtener duración del drawdown actual en segundos"""
        if self.drawdown_start:
            return int((datetime.utcnow() - self.drawdown_start).total_seconds())
        return 0

    def _record_recovery(self):
        """Registrar recuperación de drawdown"""
        if self.drawdown_start:
            event = {
                "peak": self.peak_balance,
                "trough": self.trough_balance,
                "max_drawdown_pct": self.current_drawdown_pct,
                "max_drawdown_amount": self.current_drawdown_amount,
                "start_time": self.drawdown_start,
                "end_time": datetime.utcnow(),
                "duration_seconds": self._get_drawdown_duration()
            }

            self.drawdown_events.append(event)

            logger.info(
                "drawdown_recovered",
                duration=event["duration_seconds"],
                max_drawdown=event["max_drawdown_pct"]
            )

            # Reset trough
            self.trough_balance = self.current_balance

    def get_statistics(self) -> dict:
        """Obtener estadísticas de drawdown"""
        return {
            "current": {
                "drawdown_pct": self.current_drawdown_pct,
                "drawdown_amount": self.current_drawdown_amount,
                "duration_seconds": self._get_drawdown_duration(),
                "in_drawdown": self.current_drawdown_pct > 0.1
            },
            "historical": {
                "max_drawdown_pct": self.max_drawdown_pct_historical,
                "max_drawdown_amount": self.max_drawdown_amount_historical,
                "total_events": len(self.drawdown_events)
            },
            "limits": {
                "max_drawdown_pct": self.max_drawdown_pct
            },
            "current_state": {
                "peak_balance": self.peak_balance,
                "current_balance": self.current_balance,
                "trough_balance": self.trough_balance
            }
        }