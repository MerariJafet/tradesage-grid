"""Core GridEngine implementation supporting symmetric grid generation and trailing exits."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional
from collections import deque

from .grid_config import GridConfig
from .risk_controller import RiskController, RiskLimitBreached
from .trailing_manager import TrailingManager


@dataclass(slots=True)
class GridLevel:
    index: int
    side: str
    price: float
    size: float
    filled: bool = False


@dataclass(slots=True)
class ExecutionRecord:
    timestamp: datetime
    side: str
    price: float
    size: float
    notional: float
    fee_paid: float
    pnl: float
    equity: float
    metadata: Dict[str, float]


class GridEngine:
    """Maintain a symmetric spot grid around a base price and process executions."""

    def __init__(
        self,
        config: GridConfig,
        *,
        trailing_manager: Optional[TrailingManager] = None,
        risk_controller: Optional[RiskController] = None,
    ) -> None:
        self.config = config
        self.trailing_manager = trailing_manager or TrailingManager(config.trailing_pct)
        self.risk_controller = risk_controller or RiskController(
            initial_equity=config.capital,
            max_drawdown_pct=config.max_drawdown_pct,
            max_capital_fraction=min(1.0, config.max_position / max(config.order_size, 1e-8)),
        )

        self.cash = config.capital
        self.position = 0.0
        self.realized_pnl = 0.0

        self.buy_levels: List[GridLevel] = []
        self.sell_levels: List[GridLevel] = []

        self._position_seq = 0
        self._open_positions: Deque[str] = deque()
        self._position_entry: Dict[str, float] = {}
        self._position_size: Dict[str, float] = {}

        self.generate_levels(config.base_price)

    # ---------------------------------------------------------------------
    # Level management
    # ---------------------------------------------------------------------
    def generate_levels(self, base_price: Optional[float] = None) -> None:
        base = base_price if base_price is not None else self.config.base_price
        if base <= 0:
            raise ValueError("Base price must be positive")
        self.config.base_price = base

        spacing = self.config.spacing_ratio
        self.buy_levels.clear()
        self.sell_levels.clear()

        for idx in range(1, self.config.levels + 1):
            discount = 1 - spacing * idx
            premium = 1 + spacing * idx
            buy_price = max(base * discount, 0.01)
            sell_price = base * premium
            size = self.config.order_size
            self.buy_levels.append(GridLevel(index=idx, side="buy", price=buy_price, size=size))
            self.sell_levels.append(GridLevel(index=idx, side="sell", price=sell_price, size=size))

    # ---------------------------------------------------------------------
    # Execution loop
    # ---------------------------------------------------------------------
    def update_price(self, price: float, timestamp: Optional[datetime] = None) -> List[ExecutionRecord]:
        if price <= 0:
            raise ValueError("Price must be positive")
        ts = timestamp or datetime.now(tz=timezone.utc)
        fills: List[ExecutionRecord] = []

        fills.extend(self._process_levels(self.buy_levels, price, ts))
        fills.extend(self._process_levels(self.sell_levels, price, ts))
        fills.extend(self._process_trailing(price, ts))
        return fills

    def _process_levels(self, levels: List[GridLevel], market_price: float, timestamp: datetime) -> List[ExecutionRecord]:
        fills: List[ExecutionRecord] = []
        for level in levels:
            if level.filled:
                continue
            if level.side == "buy" and market_price > level.price:
                continue
            if level.side == "sell" and market_price < level.price:
                continue

            if level.side == "sell" and self.position < level.size:
                continue

            notional = level.price * level.size
            if level.side == "buy" and not self._can_execute_buy(notional, level.size):
                continue

            try:
                record = self._execute(level, timestamp)
            except RiskLimitBreached:
                continue
            if record:
                fills.append(record)
        return fills

    def _execute(self, level: GridLevel, timestamp: datetime) -> Optional[ExecutionRecord]:
        notional = level.price * level.size
        fee_ratio = self.config.maker_fee_ratio
        fee = notional * fee_ratio

        if level.side == "buy":
            total_cost = notional + fee
            if total_cost > self.cash + 1e-9:
                return None
            self.cash -= total_cost
            self.position += level.size
            position_id = self._register_position(level)
            self.risk_controller.register_fill(notional, is_long=True)
            pnl = 0.0
        else:
            if self.position < level.size - 1e-9:
                return None
            proceeds = notional - fee
            self.cash += proceeds
            self.position -= level.size
            position_id = self._release_position()
            entry_price = self._position_entry.pop(position_id, level.price)
            self._position_size.pop(position_id, None)
            self.trailing_manager.remove(position_id)
            pnl = (level.price - entry_price) * level.size
            self.realized_pnl += pnl
            self.risk_controller.register_fill(notional, is_long=False)

        level.filled = True
        equity = self._compute_equity(level.price)
        snapshot = self.risk_controller.update_equity(equity)
        metadata = {
            "equity": snapshot.equity,
            "drawdown_pct": snapshot.drawdown_pct,
            "capital_in_use": snapshot.capital_in_use,
            "position": self.position,
        }
        return ExecutionRecord(
            timestamp=timestamp,
            side=level.side,
            price=level.price,
            size=level.size,
            notional=notional,
            fee_paid=fee,
            pnl=pnl,
            equity=equity,
            metadata=metadata,
        )

    def _process_trailing(self, price: float, timestamp: datetime) -> List[ExecutionRecord]:
        fills: List[ExecutionRecord] = []
        if not self._open_positions:
            return fills
        for position_id in list(self._open_positions):
            state = self.trailing_manager.update(position_id, price)
            if not state:
                continue
            if not self.trailing_manager.should_exit(position_id, price):
                continue
            self._open_positions.remove(position_id)
            entry_price = self._position_entry.pop(position_id, price)
            size = self._position_size.pop(position_id, self.config.order_size)
            notional = price * size
            fee = notional * self.config.taker_fee_ratio
            proceeds = notional - fee
            self.cash += proceeds
            self.position = max(self.position - size, 0.0)
            pnl = (price - entry_price) * size
            self.realized_pnl += pnl
            self.risk_controller.register_fill(notional, is_long=False)
            equity = self._compute_equity(price)
            snapshot = self.risk_controller.update_equity(equity)
            metadata = {
                "equity": snapshot.equity,
                "drawdown_pct": snapshot.drawdown_pct,
                "capital_in_use": snapshot.capital_in_use,
                "position": self.position,
                "trailing_stop": state.stop_price,
            }
            fills.append(
                ExecutionRecord(
                    timestamp=timestamp,
                    side="sell",
                    price=price,
                    size=size,
                    notional=notional,
                    fee_paid=fee,
                    pnl=pnl,
                    equity=equity,
                    metadata=metadata,
                )
            )
            self.trailing_manager.remove(position_id)
        return fills

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _register_position(self, level: GridLevel) -> str:
        self._position_seq += 1
        position_id = f"pos-{self._position_seq}"
        self._open_positions.append(position_id)
        self._position_entry[position_id] = level.price
        self._position_size[position_id] = level.size
        self.trailing_manager.register(position_id, entry_price=level.price, direction="long")
        return position_id

    def _release_position(self) -> str:
        if not self._open_positions:
            return "pos-0"
        return self._open_positions.popleft()

    def _can_execute_buy(self, notional: float, size: float) -> bool:
        if not self.risk_controller.allow_trade(notional):
            return False
        if self.position + size > self.config.max_position + 1e-9:
            return False
        required = notional * (1 + self.config.maker_fee_ratio)
        return required <= self.cash + 1e-9

    def _compute_equity(self, reference_price: float) -> float:
        return self.cash + self.position * reference_price

    # ------------------------------------------------------------------
    # Public state accessors
    # ------------------------------------------------------------------
    def state(self) -> Dict[str, float]:
        return {
            "cash": self.cash,
            "position": self.position,
            "realized_pnl": self.realized_pnl,
            "equity": self._compute_equity(self.config.base_price),
        }

    def reset(self, *, capital: Optional[float] = None, base_price: Optional[float] = None) -> None:
        if capital is not None:
            self.cash = capital
            self.config.capital = capital
        else:
            self.cash = self.config.capital
        if base_price is not None:
            self.config.base_price = base_price
        self.position = 0.0
        self.realized_pnl = 0.0
        self.buy_levels = []
        self.sell_levels = []
        self._open_positions.clear()
        self._position_entry.clear()
        self._position_size.clear()
        self._position_seq = 0
        self.generate_levels(self.config.base_price)
        self.risk_controller.reset(initial_equity=self.config.capital)
