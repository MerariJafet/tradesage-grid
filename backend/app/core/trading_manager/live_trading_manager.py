import asyncio
import datetime
import json


class LiveTradingManager:
    def __init__(self):
        self.open_positions = {}
        self.balance = 10000.0
        print('[INIT] Trading Manager started with balance:', self.balance)

    async def process_trade(self, trade):
        ts = datetime.datetime.now().isoformat()
        print(f'[TRADE] {ts} => {trade}')


if __name__ == '__main__':
    manager = LiveTradingManager()
    asyncio.run(manager.process_trade({'symbol': 'BTCUSDT', 'price': 45000}))
