"""Edge helpers for advanced regime detection, session filtering, and exit management."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def _coerce_timestamp(value: Any) -> Optional[datetime]:
    """Convert a timestamp-like value to ``datetime`` if possible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # pandas Timestamp has ``to_pydatetime``
        to_py = getattr(value, "to_pydatetime", None)
        if callable(to_py):
            return to_py()
    except Exception:
        return None
    try:
        if isinstance(value, (int, float)):
            # Assume milliseconds precision when numeric
            return datetime.utcfromtimestamp(float(value) / 1000.0)
    except Exception:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def detect_regime(bar: Dict[str, Any], adx: Optional[float], bandwidth: Optional[float]) -> str:
    """Classify current market regime using ADX and Bollinger bandwidth."""
    adx_value = adx or 0.0
    bandwidth_value = bandwidth or 0.0
    if adx_value > 25 and bandwidth_value > 0.06:
        return "tendencial"
    return "lateral"


def filter_session_liquidity(bar: Dict[str, Any], volume_sma: Optional[float],
                              session_start: int = 8, session_end: int = 16,
                              volume_multiplier: float = 1.2) -> bool:
    """Validate session window (UTC) and minimum liquidity conditions."""
    timestamp = _coerce_timestamp(bar.get("timestamp"))
    if timestamp is None:
        return False

    hour = timestamp.hour
    if hour < session_start or hour > session_end:
        return False

    if volume_sma is None:
        return False

    try:
        current_volume = float(bar.get("volume", 0.0))
    except (TypeError, ValueError):
        current_volume = 0.0

    return current_volume >= volume_multiplier * float(volume_sma)


def manage_exit(trade: Any, bar: Dict[str, Any], atr: Optional[float], adx: Optional[float]) -> bool:
    """Apply trailing-stop and breakeven logic for an open trade."""
    if trade is None:
        return False

    try:
        close_price = float(bar.get("close"))
    except (TypeError, ValueError):
        return False

    entry_price = getattr(trade, "entry_price", None)
    if entry_price is None or entry_price <= 0:
        return False

    atr_value = float(atr) if atr and atr > 0 else None
    adx_value = float(adx) if adx else None
    updated = False

    if adx_value and adx_value > 30 and atr_value:
        offset = 2.0 * atr_value
        if trade.side == "BUY":
            new_stop = close_price - offset
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
                updated = True
        elif trade.side == "SELL":
            new_stop = close_price + offset
            if new_stop < trade.stop_loss:
                trade.stop_loss = new_stop
                updated = True

    current_move = close_price - entry_price if trade.side == "BUY" else entry_price - close_price
    pnl_ratio = current_move / entry_price if entry_price else 0.0

    if pnl_ratio >= 0.005:
        if trade.side == "BUY" and trade.stop_loss < entry_price:
            trade.stop_loss = entry_price
            updated = True
        elif trade.side == "SELL" and trade.stop_loss > entry_price:
            trade.stop_loss = entry_price
            updated = True

    if updated:
        metadata = getattr(trade, "metadata", None)
        if isinstance(metadata, dict):
            actions = metadata.setdefault("management_actions", [])
            actions.append(
                {
                    "timestamp": _coerce_timestamp(bar.get("timestamp")),
                    "stop_loss": trade.stop_loss,
                    "reason": "breakeven" if pnl_ratio >= 0.005 else "trailing"
                }
            )
    return updated
