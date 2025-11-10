from typing import List, Optional, Dict
from datetime import datetime, timedelta, timezone
from app.core.risk.risk_events import RiskEvent, RiskEventType, RiskEventSeverity
from app.core.risk.drawdown_tracker import DrawdownTracker
from app.core.risk.position_limits import PositionLimits
from app.utils.logger import get_logger

logger = get_logger("risk_manager")

class RiskManager:
    """
    Risk Manager Principal

    Responsabilidades:
    - Monitorear límites de pérdida (diaria, semanal, mensual)
    - Tracking de drawdowns
    - Límites de posición
    - Detección de patrones de pérdida (consecutive losses)
    - Kill-switch automático
    - Cooldown periods
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_daily_loss_pct: float = 2.0,      # 2% max pérdida diaria
        max_weekly_loss_pct: float = 5.0,     # 5% max pérdida semanal
        max_drawdown_pct: float = 10.0,       # 10% max drawdown
        max_consecutive_losses: int = 3,       # 3 pérdidas consecutivas
        cooldown_after_loss_minutes: int = 15  # 15 min cooldown después de pérdida
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Límites
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_after_loss_minutes = cooldown_after_loss_minutes

        # Componentes
        self.drawdown_tracker = DrawdownTracker(
            initial_balance=initial_balance,
            max_drawdown_pct=max_drawdown_pct
        )

        self.position_limits = PositionLimits(
            account_balance=initial_balance
        )

        # Estado
        self.trading_enabled = True
        self.kill_switch_active = False
        self.kill_switch_reason = None

        # Tracking de pérdidas
        self.consecutive_losses = 0
        self.last_trade_result = None
        self.last_loss_timestamp = None

        # Pérdidas por período
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.day_start_balance = initial_balance
        self.week_start_balance = initial_balance
        self.current_day = datetime.now(timezone.utc).date()
        self.current_week = datetime.now(timezone.utc).isocalendar()[1]

        # Eventos de riesgo
        self.risk_events: List[RiskEvent] = []

        logger.info(
            "risk_manager_initialized",
            initial_balance=initial_balance,
            max_daily_loss_pct=max_daily_loss_pct,
            max_weekly_loss_pct=max_weekly_loss_pct,
            max_drawdown_pct=max_drawdown_pct
        )

    def update_balance(self, new_balance: float) -> List[RiskEvent]:
        """
        Actualizar balance y verificar límites

        Returns:
            Lista de eventos de riesgo generados
        """
        self.current_balance = new_balance
        events = []

        # Actualizar drawdown tracker
        drawdown_event = self.drawdown_tracker.update(new_balance)
        if drawdown_event:
            if drawdown_event["type"] == "max_drawdown_exceeded":
                event = RiskEvent(
                    type=RiskEventType.MAX_DRAWDOWN,
                    severity=RiskEventSeverity.CRITICAL,
                    message=f"Max drawdown exceeded: {drawdown_event['current_drawdown_pct']:.2f}%",
                    value=drawdown_event['current_drawdown_pct'],
                    limit=drawdown_event['limit']
                )
                events.append(event)
                self._trigger_kill_switch(event)

            elif drawdown_event["type"] == "drawdown_warning":
                event = RiskEvent(
                    type=RiskEventType.MAX_DRAWDOWN,
                    severity=RiskEventSeverity.WARNING,
                    message=f"Approaching max drawdown: {drawdown_event['current_drawdown_pct']:.2f}%",
                    value=drawdown_event['current_drawdown_pct'],
                    limit=drawdown_event['limit']
                )
                events.append(event)

        # Actualizar position limits
        self.position_limits.update_balance(new_balance)

        # Verificar límites de período
        self._check_period_resets()
        self._update_period_pnl(new_balance)

        # Verificar límites diarios
        daily_loss_pct = abs(self.daily_pnl / self.day_start_balance * 100) if self.day_start_balance > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
            event = RiskEvent(
                type=RiskEventType.DAILY_LOSS_LIMIT,
                severity=RiskEventSeverity.CRITICAL,
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2f}%",
                value=daily_loss_pct,
                limit=self.max_daily_loss_pct
            )
            events.append(event)
            self._trigger_kill_switch(event)

        # Verificar límites semanales
        weekly_loss_pct = abs(self.weekly_pnl / self.week_start_balance * 100) if self.week_start_balance > 0 else 0
        if self.weekly_pnl < 0 and weekly_loss_pct >= self.max_weekly_loss_pct:
            event = RiskEvent(
                type=RiskEventType.WEEKLY_LOSS_LIMIT,
                severity=RiskEventSeverity.CRITICAL,
                message=f"Weekly loss limit exceeded: {weekly_loss_pct:.2f}%",
                value=weekly_loss_pct,
                limit=self.max_weekly_loss_pct
            )
            events.append(event)
            self._trigger_kill_switch(event)

        # Guardar eventos
        self.risk_events.extend(events)

        return events

    def register_trade_result(
        self,
        pnl: float,
        symbol: str,
        strategy: str
    ) -> List[RiskEvent]:
        """
        Registrar resultado de trade

        Returns:
            Lista de eventos de riesgo generados
        """
        events = []

        # Tracking de pérdidas consecutivas
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_timestamp = datetime.now(timezone.utc)

            if self.consecutive_losses >= self.max_consecutive_losses:
                event = RiskEvent(
                    type=RiskEventType.CONSECUTIVE_LOSSES,
                    severity=RiskEventSeverity.WARNING,
                    message=f"Consecutive losses: {self.consecutive_losses}",
                    value=self.consecutive_losses,
                    limit=self.max_consecutive_losses,
                    symbol=symbol,
                    strategy=strategy
                )
                events.append(event)

                # Activar cooldown
                self.trading_enabled = False
                logger.warning(
                    "cooldown_activated",
                    consecutive_losses=self.consecutive_losses,
                    cooldown_minutes=self.cooldown_after_loss_minutes
                )
        else:
            # Reset consecutive losses en profit
            self.consecutive_losses = 0

        self.last_trade_result = "loss" if pnl < 0 else "profit"

        # Guardar eventos
        self.risk_events.extend(events)

        return events

    def can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Verificar si se puede tradear

        Returns:
            (can_trade, reason_if_not)
        """

        # Kill-switch activo
        if self.kill_switch_active:
            return False, f"Kill-switch active: {self.kill_switch_reason}"

        # Trading deshabilitado
        if not self.trading_enabled:
            # Verificar si el cooldown expiró
            if self.last_loss_timestamp:
                cooldown_end = self.last_loss_timestamp + timedelta(minutes=self.cooldown_after_loss_minutes)
                if datetime.now(timezone.utc) < cooldown_end:
                    remaining = (cooldown_end - datetime.now(timezone.utc)).total_seconds() / 60
                    return False, f"Cooldown active: {remaining:.1f} minutes remaining"
                else:
                    # Cooldown expiró, reactivar
                    self.trading_enabled = True
                    self.consecutive_losses = 0
                    logger.info("cooldown_expired", trading_reenabled=True)

        return True, None

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

        # Verificar si trading está habilitado
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason

        # Verificar límites de posición
        return self.position_limits.can_open_position(symbol, position_size, risk_amount)

    def register_position_opened(
        self,
        position_id: str,
        symbol: str,
        risk_amount: float
    ):
        """Registrar apertura de posición"""
        self.position_limits.register_position_opened(position_id, symbol, risk_amount)

    def register_position_closed(
        self,
        position_id: str,
        symbol: str,
        risk_amount: float
    ):
        """Registrar cierre de posición"""
        self.position_limits.register_position_closed(position_id, symbol, risk_amount)

    def _trigger_kill_switch(self, event: RiskEvent):
        """Activar kill-switch"""
        self.kill_switch_active = True
        self.trading_enabled = False
        self.kill_switch_reason = event.message

        logger.critical(
            "kill_switch_activated",
            reason=event.message,
            type=event.type,
            value=event.value,
            limit=event.limit
        )

    def manual_kill_switch(self, reason: str = "Manual emergency stop"):
        """Activar kill-switch manualmente"""
        event = RiskEvent(
            type=RiskEventType.EMERGENCY_STOP,
            severity=RiskEventSeverity.EMERGENCY,
            message=reason,
            value=0,
            limit=0
        )
        self._trigger_kill_switch(event)
        self.risk_events.append(event)

    def reset_kill_switch(self):
        """Desactivar kill-switch (solo manual)"""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.trading_enabled = True
        self.consecutive_losses = 0

        logger.warning("kill_switch_reset", manual=True)

    def _check_period_resets(self):
        """Verificar si hay que resetear contadores de período"""
        now = datetime.now(timezone.utc)

        # Reset diario
        if now.date() != self.current_day:
            self.current_day = now.date()
            self.day_start_balance = self.current_balance
            self.daily_pnl = 0.0
            logger.info("daily_reset", date=str(self.current_day))

        # Reset semanal
        current_week = now.isocalendar()[1]
        if current_week != self.current_week:
            self.current_week = current_week
            self.week_start_balance = self.current_balance
            self.weekly_pnl = 0.0
            logger.info("weekly_reset", week=self.current_week)

    def _update_period_pnl(self, new_balance: float):
        """Actualizar PnL de períodos"""
        self.daily_pnl = new_balance - self.day_start_balance
        self.weekly_pnl = new_balance - self.week_start_balance

    def get_statistics(self) -> dict:
        """Obtener estadísticas completas del risk manager"""
        return {
            "status": {
                "trading_enabled": self.trading_enabled,
                "kill_switch_active": self.kill_switch_active,
                "kill_switch_reason": self.kill_switch_reason,
                "consecutive_losses": self.consecutive_losses,
                "in_cooldown": not self.trading_enabled and not self.kill_switch_active
            },
            "balance": {
                "current": self.current_balance,
                "initial": self.initial_balance,
                "total_pnl": self.current_balance - self.initial_balance,
                "total_pnl_pct": ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
            },
            "period_pnl": {
                "daily": {
                    "pnl": self.daily_pnl,
                    "pnl_pct": (self.daily_pnl / self.day_start_balance * 100) if self.day_start_balance > 0 else 0,
                    "limit_pct": self.max_daily_loss_pct
                },
                "weekly": {
                    "pnl": self.weekly_pnl,
                    "pnl_pct": (self.weekly_pnl / self.week_start_balance * 100) if self.week_start_balance > 0 else 0,
                    "limit_pct": self.max_weekly_loss_pct
                }
            },
            "drawdown": self.drawdown_tracker.get_statistics(),
            "position_limits": self.position_limits.get_statistics(),
            "risk_events": [e.to_dict() for e in self.risk_events[-10:]]  # Últimos 10 eventos
        }