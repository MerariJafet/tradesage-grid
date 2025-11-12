"""Quick comparative visualization helpers for Sprint 9B."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend suitable for CI/servers.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

from backend.app.backtesting.comparative_engine import ComparativeBacktester

OUTPUT_PATH = Path("reports/comparative_pnl_bar.png")


def plot_comparative_curves(output_path: Path = OUTPUT_PATH) -> None:
    """Render a basic bar chart comparing adaptive PnL vs ML signal strength."""

    if plt is None:
        print("[PLOT] matplotlib not available; install it to generate comparative plots.")
        return

    engine = ComparativeBacktester()
    report = engine.run_backtest() or {}
    if not report:
        print("[PLOT] No comparative data available yet; skipping plot generation.")
        return

    df = pd.DataFrame([report])

    fig, ax = plt.subplots(figsize=(6, 4))
    df[["adaptive_pnl", "ml_signal_mean"]].plot(
        kind="bar",
        ax=ax,
        color=["#1f77b4", "#ff7f0e"],
    )
    ax.set_title("ML vs Adaptive Comparative Performance")
    ax.set_ylabel("Normalized Values")
    ax.set_xticklabels(["Snapshot"], rotation=0)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Saved {output_path}")


if __name__ == "__main__":  # pragma: no cover
    plot_comparative_curves()
