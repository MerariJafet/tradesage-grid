"""Microstructure signal helpers."""
from __future__ import annotations

from typing import Optional


def microstructure_signal(bar, bid_vol: float, ask_vol: float, funding_rate: float) -> Optional[str]:
    """Infer directional bias from order-book imbalance and funding."""
    try:
        bid_volume = float(bid_vol)
        ask_volume = float(ask_vol)
    except (TypeError, ValueError):
        return None

    if ask_volume <= 0:
        imbalance = float("inf")
    else:
        imbalance = bid_volume / ask_volume

    funding = float(funding_rate or 0.0)

    if imbalance > 1.5 and funding < 0.01:
        return "buy"
    if imbalance < 0.67 and funding > 0.01:
        return "sell"
    return None
