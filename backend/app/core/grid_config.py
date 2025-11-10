"""Configuration primitives for the Grid + Trailing execution engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class GridConfig:
    """Typed container for grid engine parameters.

    spacing_pct is expressed in percentage points (e.g. 0.25 == 0.25%).
    """

    symbol: str = "BTCUSDT"
    base_price: float = 50000.0
    spacing_pct: float = 0.25
    levels: int = 6
    order_size: float = 0.001
    capital: float = 10000.0
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04
    trailing_pct: float = 0.5
    max_drawdown_pct: float = 25.0
    max_position: float = field(default=0.02)

    def __post_init__(self) -> None:
        if self.levels <= 0:
            raise ValueError("levels must be > 0")
        if self.spacing_pct <= 0:
            raise ValueError("spacing_pct must be > 0")
        if self.order_size <= 0:
            raise ValueError("order_size must be > 0")
        if self.capital <= 0:
            raise ValueError("capital must be > 0")

    @property
    def spacing_ratio(self) -> float:
        """Return spacing as decimal ratio (0.25% -> 0.0025)."""

        return self.spacing_pct / 100.0

    @property
    def maker_fee_ratio(self) -> float:
        return self.maker_fee_pct / 100.0

    @property
    def taker_fee_ratio(self) -> float:
        return self.taker_fee_pct / 100.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_price": self.base_price,
            "spacing_pct": self.spacing_pct,
            "levels": self.levels,
            "order_size": self.order_size,
            "capital": self.capital,
            "maker_fee_pct": self.maker_fee_pct,
            "taker_fee_pct": self.taker_fee_pct,
            "trailing_pct": self.trailing_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_position": self.max_position,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GridConfig":
        """Instantiate a configuration object from a dictionary."""

        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path) -> "GridConfig":
        import json

        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
