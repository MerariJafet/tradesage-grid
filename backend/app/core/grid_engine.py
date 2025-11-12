"""Core GridEngine implementation supporting symmetric grid generation and trailing exits."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, TYPE_CHECKING

from .grid_config import GridConfig
from .risk_controller import RiskController, RiskLimitBreached
from .trailing_manager import TrailingManager

if TYPE_CHECKING:  # pragma: no cover
    from ..ml.signal_model import SignalModel, SignalPrediction


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
        signal_model: Optional["SignalModel"] = None,
    ) -> None:
        self.config = config
        self.cfg = config
        self.trailing_manager = trailing_manager or TrailingManager(config.trailing_pct)
        self.risk_controller = risk_controller or RiskController(
            initial_equity=config.capital,
            max_drawdown_pct=config.max_drawdown_pct,
            max_capital_fraction=min(1.0, config.max_position / max(config.order_size, 1e-8)),
            config=config,
        )
        self.signal_model: Optional["SignalModel"] = signal_model
        if self.signal_model is None and self.config.ml_enabled:
            try:
                from ..ml.signal_model import SignalModel as _SignalModel

                self.signal_model = _SignalModel.load_from_disk(
                    mode=self.config.ml_mode,
                    threshold=self.config.ml_threshold,
                    window=self.config.ml_window,
                    horizon=self.config.ml_horizon,
                )
            except FileNotFoundError:
                self.signal_model = None
        self._ml_active = self.config.ml_enabled and self.signal_model is not None
        self._price_history: Deque[float] = deque(maxlen=self.config.ml_window + self.config.ml_horizon + 5)
        self._last_signal: Optional["SignalPrediction"] = None
        self._base_spacing_pct = config.spacing_pct
        self._base_levels = config.levels
        self._base_trailing_pct = config.trailing_pct
        self._current_confidence_band: Optional[str] = None
        self._gate_reason = "n/a"
        self._last_prob_up: Optional[float] = None
        self._last_prob_down: Optional[float] = None
        self._bar_index = 0
        self._last_gate_log_bar = -1
        self._last_gate_log_reason = ""
        self._last_gate_log_event = ""

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
        current_bar = self._bar_index
        self._price_history.append(price)
        self._last_signal = self._evaluate_signal()
        self._apply_adaptive_changes()

        fills.extend(self._process_levels(self.buy_levels, price, ts, current_bar))
        fills.extend(self._process_levels(self.sell_levels, price, ts, current_bar))
        fills.extend(self._process_trailing(price, ts, current_bar))
        self._bar_index += 1
        return fills

    def _process_levels(
        self,
        levels: List[GridLevel],
        market_price: float,
        timestamp: datetime,
        current_bar: int,
    ) -> List[ExecutionRecord]:
        fills: List[ExecutionRecord] = []
        cooldown_logged = False
        for level in levels:
            if level.filled:
                continue
            if level.side == "buy" and market_price > level.price:
                continue
            if level.side == "sell" and market_price < level.price:
                continue
            if level.side == "buy":
                if hasattr(self.risk_controller, "should_cooldown") and self.risk_controller.should_cooldown(current_bar):
                    if not cooldown_logged:
                        self._log_gate_event("cooldown", self._last_prob_up or 0.0, "cooldown", current_bar)
                        cooldown_logged = True
                    continue
                if self._ml_active and not self._ml_allows_entry(current_bar):
                    continue

            if level.side == "sell" and self.position < level.size:
                continue

            notional = level.price * level.size
            if level.side == "buy" and not self._can_execute_buy(notional, level.size):
                continue

            try:
                record = self._execute(level, timestamp, current_bar)
            except RiskLimitBreached:
                continue
            if record:
                fills.append(record)
        return fills

    def _execute(self, level: GridLevel, timestamp: datetime, current_bar: int) -> Optional[ExecutionRecord]:
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
            if hasattr(self.risk_controller, "record_trade"):
                self.risk_controller.record_trade(pnl, current_bar)

        level.filled = True
        equity = self._compute_equity(level.price)
        snapshot = self.risk_controller.update_equity(equity)
        metadata = {
            "equity": snapshot.equity,
            "drawdown_pct": snapshot.drawdown_pct,
            "capital_in_use": snapshot.capital_in_use,
            "position": self.position,
        }
        if self._last_signal is not None:
            prob_up = self._last_prob_up if self._last_prob_up is not None else self._last_signal.probability
            prob_down = self._last_prob_down if self._last_prob_down is not None else max(0.0, 1.0 - prob_up)
            metadata.update(
                {
                    "ml_probability": prob_up,
                    "ml_probability_down": prob_down,
                    "ml_decision": float(self._last_signal.decision),
                }
            )
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

    def _process_trailing(self, price: float, timestamp: datetime, current_bar: int) -> List[ExecutionRecord]:
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
            if hasattr(self.risk_controller, "record_trade"):
                self.risk_controller.record_trade(pnl, current_bar)
            equity = self._compute_equity(price)
            snapshot = self.risk_controller.update_equity(equity)
            metadata = {
                "equity": snapshot.equity,
                "drawdown_pct": snapshot.drawdown_pct,
                "capital_in_use": snapshot.capital_in_use,
                "position": self.position,
                "trailing_stop": state.stop_price,
            }
            if self._last_signal is not None:
                prob_up = self._last_prob_up if self._last_prob_up is not None else self._last_signal.probability
                prob_down = self._last_prob_down if self._last_prob_down is not None else max(0.0, 1.0 - prob_up)
                metadata.update(
                    {
                        "ml_probability": prob_up,
                        "ml_probability_down": prob_down,
                        "ml_decision": float(self._last_signal.decision),
                    }
                )
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

    def _evaluate_signal(self) -> Optional["SignalPrediction"]:
        if not self._ml_active:
            self._last_prob_up = None
            self._last_prob_down = None
            return None
        if self.signal_model is None:
            self._last_prob_up = None
            self._last_prob_down = None
            return None
        prices = list(self._price_history)
        try:
            prediction = self.signal_model.predict(prices)
        except RuntimeError:
            self._last_prob_up = None
            self._last_prob_down = None
            return None
        self._last_prob_up = float(prediction.probability)
        self._last_prob_down = max(0.0, 1.0 - self._last_prob_up)
        return prediction

    def _ml_allows_entry(self, current_bar: int) -> bool:
        if not self._ml_active:
            self._gate_reason = "ml_disabled"
            return True
        if self._last_signal is None:
            self._gate_reason = "no_signal"
            self._log_gate_event("gate", self._last_prob_up or 0.0, self._gate_reason, current_bar)
            return False
        prob_up = self._last_prob_up if self._last_prob_up is not None else self._last_signal.probability
        prob_down = self._last_prob_down if self._last_prob_down is not None else max(0.0, 1.0 - prob_up)
        if not self.evaluate_ml_gate(prob_up, prob_down):
            self._log_gate_event("gate", prob_up, self._gate_reason, current_bar)
            return False
        decision = self._last_signal.decision
        if not decision and self.config.ml_mode == "probability":
            self._gate_reason = "model_reject"
            self._log_gate_event("gate", prob_up, self._gate_reason, current_bar)
            return False
        self._gate_reason = "ok"
        return True

    def evaluate_ml_gate(self, ml_prob_up: float, ml_prob_down: float) -> bool:
        _ = ml_prob_down  # placeholder for future asymmetric gating support
        edge = abs(ml_prob_up - 0.5)
        min_edge = getattr(self.config, "ml_min_edge", 0.12)
        threshold = getattr(self.config, "ml_threshold", 0.55)
        if edge < min_edge:
            self._gate_reason = "edge_fail"
            return False
        if ml_prob_up < threshold:
            self._gate_reason = "threshold_fail"
            return False
        self._gate_reason = "ok"
        return True

    # ------------------------------------------------------------------
    # Adaptive grid helpers
    # ------------------------------------------------------------------
    def _apply_adaptive_changes(self) -> None:
        if not self.config.adaptive_mode:
            return
        if self._last_signal is None:
            return
        probability = self._last_signal.probability
        self.adjust_grid_by_confidence(probability)

    def adjust_grid_by_confidence(self, probability: float) -> None:
        band = self._select_confidence_band(probability)
        if band is None:
            if self._current_confidence_band is None:
                return
            self.config.spacing_pct = self._base_spacing_pct
            self.config.levels = self._base_levels
            self.config.trailing_pct = self._base_trailing_pct
            self.trailing_manager.update_default(self._base_trailing_pct)
            self.generate_levels(self.config.base_price)
            self._current_confidence_band = None
            self._log_adaptive_change(probability, "base")
            return
        if band == self._current_confidence_band:
            return

        spacing_map = self.config.adaptive_multipliers.get("spacing", {})
        level_map = self.config.adaptive_multipliers.get("levels", {})
        trailing_map = self.config.adaptive_multipliers.get("trailing", {})

        spacing_multiplier = spacing_map.get(band, 1.0)
        target_levels = level_map.get(band, float(self._base_levels))
        target_trailing = trailing_map.get(band, self._base_trailing_pct)

        new_spacing = max(self._base_spacing_pct * spacing_multiplier, 0.01)
        new_levels = max(int(round(target_levels)), 1)
        new_trailing = max(float(target_trailing), 0.01)

        spacing_changed = not math.isclose(self.config.spacing_pct, new_spacing, rel_tol=1e-6)
        levels_changed = self.config.levels != new_levels
        trailing_changed = not math.isclose(self.config.trailing_pct, new_trailing, rel_tol=1e-6)

        if not any([spacing_changed, levels_changed, trailing_changed]):
            self._current_confidence_band = band
            return

        self.config.spacing_pct = new_spacing
        self.config.levels = new_levels
        self.config.trailing_pct = new_trailing
        self.trailing_manager.update_default(new_trailing)
        self.generate_levels(self.config.base_price)
        self._current_confidence_band = band
        self._log_adaptive_change(probability, band)

    def _select_confidence_band(self, probability: float) -> Optional[str]:
        bands = self.config.ml_confidence_bands
        low = bands.get("low", 0.0)
        neutral = bands.get("neutral", low)
        high = bands.get("high", 1.0)
        if probability >= high:
            return "high"
        if probability >= neutral:
            return "neutral"
        if probability >= low:
            return "low"
        return None

    def _log_adaptive_change(self, probability: float, band: str) -> None:
        log_path = Path("logs/ml_adaptive_changes.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        line = (
            f"{timestamp} symbol={self.config.symbol} band={band} prob={probability:.4f} "
            f"spacing_pct={self.config.spacing_pct:.6f} levels={self.config.levels} "
            f"trailing_pct={self.config.trailing_pct:.6f}\n"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _log_gate_event(self, event: str, probability: float, reason: str, current_bar: int) -> None:
        log_path = Path("logs/ml_adaptive_changes.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            self._last_gate_log_bar == current_bar
            and self._last_gate_log_reason == reason
            and self._last_gate_log_event == event
        ):
            return
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        band = self._current_confidence_band or "base"
        line = (
            f"{timestamp} symbol={self.config.symbol} event={event} reason={reason} "
            f"prob_up={probability:.4f} band={band}\n"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self._last_gate_log_bar = current_bar
        self._last_gate_log_reason = reason
        self._last_gate_log_event = event

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
        self._bar_index = 0
        self._last_signal = None
        self._last_prob_up = None
        self._last_prob_down = None
        self._gate_reason = "n/a"
        self._last_gate_log_bar = -1
        self._last_gate_log_reason = ""
        self._last_gate_log_event = ""
