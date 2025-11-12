"""Simple capital and drawdown risk guardrails for the grid engine."""
from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Any, Optional


class RiskLimitBreached(RuntimeError):
    """Raised when a safety rail is exceeded."""


@dataclass(slots=True)
class RiskSnapshot:
    equity: float
    drawdown_pct: float
    capital_in_use: float


class RiskController:
    """Tracks basic drawdown and capital usage constraints."""

    def __init__(
        self,
        *,
        initial_equity: Optional[float] = None,
        max_drawdown_pct: float = 25.0,
        max_capital_fraction: float = 1.0,
        config: Optional[Any] = None,
        max_exposure: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        cooldown_time: Optional[float] = None,
    ) -> None:
        if initial_equity is None:
            initial_equity = 0.0
        self.initial_equity = initial_equity
        self.max_drawdown_pct = max_drawdown_pct
        self.max_capital_fraction = max(0.0, min(1.0, max_capital_fraction))
        self.config = config
        self.cfg = config

        cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else cooldown_time
        if cooldown_seconds is not None and cooldown_seconds < 0:
            cooldown_seconds = 0.0
        self.cooldown_seconds = cooldown_seconds
        if max_exposure is not None and max_exposure < 0:
            max_exposure = 0.0
        self.max_exposure_fraction = max_exposure if max_exposure is not None else self.max_capital_fraction

        self._equity_peak = initial_equity
        self._equity = initial_equity
        self._capital_in_use = 0.0
        self._loss_streak = 0
        self._last_loss_bar = -1
        self._last_trade_time: Optional[datetime] = None

        if self.cooldown_seconds is not None or max_exposure is not None:
            print(
                f"[RISK] Controller ready: max_exposure {self.max_exposure_fraction:.2%} "
                f"cooldown={self.cooldown_seconds if self.cooldown_seconds is not None else 'disabled'}"
            )

    def allow_trade(self, notional: float) -> bool:
        """Return True when the trade keeps capital usage within bounds."""

        if notional <= 0:
            return False
        projected = self._capital_in_use + notional
        limit = self.initial_equity * self.max_capital_fraction
        return projected <= limit + 1e-8

    def register_fill(self, notional: float, is_long: bool) -> None:
        """Update capital in use based on executed fill."""

        delta = notional if is_long else -notional
        self._capital_in_use = max(self._capital_in_use + delta, 0.0)

    def update_equity(self, equity: float) -> RiskSnapshot:
        self._equity = equity
        if equity > self._equity_peak:
            self._equity_peak = equity
        drawdown = self.drawdown_pct
        if drawdown > self.max_drawdown_pct:
            raise RiskLimitBreached(
                f"Drawdown {drawdown:.2f}% exceeds limit {self.max_drawdown_pct:.2f}%"
            )
        return RiskSnapshot(equity=equity, drawdown_pct=drawdown, capital_in_use=self._capital_in_use)

    def record_trade(
        self,
        trade_result: float,
        current_bar: Optional[int] = None,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.now()
        self._last_trade_time = timestamp

        if trade_result < 0:
            self._loss_streak += 1
            if current_bar is not None:
                self._last_loss_bar = current_bar
        elif trade_result > 0:
            self._loss_streak = 0

        if current_bar is None:
            print(f"[RISK] Trade recorded | PnL: {trade_result:.2f} | Loss streak: {self._loss_streak}")

    def should_cooldown(self, current_bar: Optional[int] = None) -> bool:
        if self.cooldown_seconds is not None:
            if self._last_trade_time is None:
                return False
            elapsed = (datetime.now() - self._last_trade_time).total_seconds()
            return elapsed < self.cooldown_seconds

        if current_bar is None:
            return False
        cooldown_after = getattr(self.config, "cooldown_after_losses", 3)
        cooldown_bars = getattr(self.config, "cooldown_bars", 15)
        if cooldown_after <= 0 or cooldown_bars <= 0:
            return False
        if self._loss_streak < cooldown_after:
            return False
        if self._last_loss_bar < 0:
            return False
        return (current_bar - self._last_loss_bar) <= cooldown_bars

    def check_exposure(self, balance: float, open_positions_value: float) -> bool:
        if balance <= 0:
            print("[RISK] Exposure check failed: balance is non-positive")
            return False
        if self.max_exposure_fraction <= 0:
            return True
        exposure = open_positions_value / balance
        if exposure > self.max_exposure_fraction:
            print(f"[RISK] Exposure limit reached: {exposure:.2%}")
            return False
        return True

    @property
    def drawdown_pct(self) -> float:
        if self._equity_peak <= 0:
            return 0.0
        return max(0.0, (1 - self._equity / self._equity_peak) * 100.0)

    @property
    def capital_in_use(self) -> float:
        return self._capital_in_use

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def equity_peak(self) -> float:
        return self._equity_peak

    def snapshot(self) -> RiskSnapshot:
        return RiskSnapshot(
            equity=self._equity,
            drawdown_pct=self.drawdown_pct,
            capital_in_use=self._capital_in_use,
        )

    def reset(self, initial_equity: Optional[float] = None) -> None:
        if initial_equity is not None:
            self.initial_equity = initial_equity
            self._equity = initial_equity
            self._equity_peak = initial_equity
        else:
            self._equity = self.initial_equity
            self._equity_peak = self.initial_equity
        self._capital_in_use = 0.0
        self._loss_streak = 0
        self._last_loss_bar = -1
