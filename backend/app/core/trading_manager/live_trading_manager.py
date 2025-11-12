import asyncio
import datetime


class LiveTradingManager:
    def __init__(self):
        self.open_positions = {}
        self.balance = 10000.0
        self.total_volume = 0.0
        self.avg_price = 0.0
        print("[INIT] Trading Manager active with balance:", self.balance)

    async def process_trade(self, trade: dict) -> None:
        qty = float(trade.get("qty", 0.0))
        price = float(trade.get("price", 0.0))
        if qty <= 0 or price <= 0:
            return
        self.total_volume += qty
        if self.avg_price > 0:
            self.avg_price = (self.avg_price + price) / 2
        else:
            self.avg_price = price
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(
            f"[TRADE] {ts} | {trade.get('symbol', 'UNKNOWN')} | {price:.2f} "
            f"| Qty: {qty:.6f} | Avg: {self.avg_price:.2f} | Vol: {self.total_volume:.6f}"
        )


if __name__ == "__main__":
    manager = LiveTradingManager()
    asyncio.run(manager.process_trade({"symbol": "BTCUSDT", "price": 45000, "qty": 0.05}))
