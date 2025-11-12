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
    ml_enabled: bool = False
    ml_mode: str = "probability"
    ml_threshold: float = 0.55
    ml_window: int = 60
    ml_horizon: int = 3
    adaptive_mode: bool = True
    ml_confidence_bands: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.45, "mid": 0.6, "high": 0.75}
    )
    adaptive_multipliers: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "spacing": {"low": 1.5, "mid": 1.0, "high": 0.7},
            "levels": {"low": 5.0, "mid": 10.0, "high": 15.0},
            "trailing": {"low": 0.3, "mid": 0.5, "high": 0.7},
        }
    )

    def __post_init__(self) -> None:
        if self.levels <= 0:
            raise ValueError("levels must be > 0")
        if self.spacing_pct <= 0:
            raise ValueError("spacing_pct must be > 0")
        if self.order_size <= 0:
            raise ValueError("order_size must be > 0")
        if self.capital <= 0:
            raise ValueError("capital must be > 0")
        if self.ml_threshold <= 0 or self.ml_threshold >= 1:
            raise ValueError("ml_threshold must be between 0 and 1")
        if self.ml_window <= 1:
            raise ValueError("ml_window must be > 1")
        if self.ml_horizon <= 0:
            raise ValueError("ml_horizon must be > 0")
        if self.ml_mode not in {"probability", "binary"}:
            raise ValueError("ml_mode must be 'probability' or 'binary'")
        self._validate_adaptive_settings()

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
            "ml_enabled": self.ml_enabled,
            "ml_mode": self.ml_mode,
            "ml_threshold": self.ml_threshold,
            "ml_window": self.ml_window,
            "ml_horizon": self.ml_horizon,
            "adaptive_mode": self.adaptive_mode,
            "ml_confidence_bands": self.ml_confidence_bands,
            "adaptive_multipliers": self.adaptive_multipliers,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_adaptive_settings(self) -> None:
        required_bands = {"low", "mid", "high"}
        if not required_bands.issubset(self.ml_confidence_bands):
            raise ValueError("ml_confidence_bands must define low, mid, and high thresholds")

        bands = self.ml_confidence_bands
        low, mid, high = bands["low"], bands["mid"], bands["high"]
        if not (0.0 < low < mid < high < 1.0):
            raise ValueError("ml_confidence_bands thresholds must satisfy 0 < low < mid < high < 1")

        if not isinstance(self.adaptive_multipliers, dict):
            raise ValueError("adaptive_multipliers must be a dictionary")

        required_sections = {"spacing", "levels", "trailing"}
        if not required_sections.issubset(self.adaptive_multipliers):
            raise ValueError("adaptive_multipliers must include spacing, levels, and trailing keys")

        for section in required_sections:
            values = self.adaptive_multipliers.get(section, {})
            if not required_bands.issubset(values):
                raise ValueError(f"adaptive_multipliers['{section}'] must define low, mid, and high entries")
            for band_key, value in values.items():
                if section == "levels":
                    if value <= 0:
                        raise ValueError("adaptive levels must be positive")
                else:
                    if value <= 0:
                        raise ValueError("adaptive multipliers must be positive")
