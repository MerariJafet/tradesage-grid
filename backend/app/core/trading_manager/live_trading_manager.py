import asyncio
import datetime
import sys
from pathlib import Path

try:
    from backend.app.core.risk_controller import RiskController
    from backend.app.core.persistence.trade_logger import TradeLogger
except ModuleNotFoundError:  # pragma: no cover - runtime fallback for script execution
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from backend.app.core.risk_controller import RiskController
    from backend.app.core.persistence.trade_logger import TradeLogger


class LiveTradingManager:
    def __init__(self):
        self.balance = 10000.0
        self.total_volume = 0.0
        self.avg_price = 0.0
        self.risk = RiskController(initial_equity=self.balance, max_exposure=0.20, cooldown_seconds=10.0)
        self.logger = TradeLogger()
        print("[INIT] Trading Manager active with persistence enabled. Balance:", self.balance)

    async def process_trade(self, trade: dict) -> None:
        if self.risk.should_cooldown():
            print("[COOLDOWN] Skipping trade due to cooldown period.")
            return

        qty = float(trade.get("qty", 0.0))
        price = float(trade.get("price", 0.0))
        if qty <= 0 or price <= 0:
            return

        open_value = (self.total_volume + qty) * price
        if not self.risk.check_exposure(self.balance, open_value):
            print("[BLOCKED] Exposure too high — trade skipped.")
            return

        pnl = (price - self.avg_price) * qty if self.total_volume > 0 else 0.0
        timestamp = datetime.datetime.now()
        self.risk.record_trade(pnl, timestamp=timestamp)

        previous_volume = self.total_volume
        new_volume = previous_volume + qty
        if previous_volume <= 0:
            self.avg_price = price
        else:
            self.avg_price = ((self.avg_price * previous_volume) + (price * qty)) / new_volume
        self.total_volume = new_volume

        ts = timestamp.strftime("%H:%M:%S")
        print(
            f"[TRADE] {ts} | {trade.get('symbol', 'UNKNOWN')} | {price:.2f} "
            f"| Qty: {qty:.6f} | Avg: {self.avg_price:.2f} | Vol: {self.total_volume:.6f}"
        )
        exposure = (self.total_volume * price / self.balance) if self.balance > 0 else 0.0
        self.logger.log_trade(
            trade.get("symbol", "UNKNOWN"),
            price,
            qty,
            pnl,
            exposure,
            timestamp=timestamp.isoformat(),
        )


if __name__ == "__main__":
    manager = LiveTradingManager()
    asyncio.run(manager.process_trade({"symbol": "BTCUSDT", "price": 45000, "qty": 0.05}))
