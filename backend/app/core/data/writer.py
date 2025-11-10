from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.config import settings
from app.utils.logger import get_logger
from typing import List, Dict
import asyncio
from datetime import datetime

logger = get_logger("data_writer")

class DataWriter:
    def __init__(self):
        self.engine = create_async_engine(settings.DATABASE_URL, echo=False)
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Buffers para batch inserts
        self.tick_buffer: List[Dict] = []
        self.bar_buffer: List[Dict] = []
        self.orderbook_buffer: List[Dict] = []

        # Configuración de batching
        self.batch_size = 100
        self.flush_interval = 1.0  # segundos

        # Métricas
        self.ticks_written = 0
        self.bars_written = 0
        self.orderbooks_written = 0

        # Locks
        self.tick_lock = asyncio.Lock()
        self.bar_lock = asyncio.Lock()
        self.ob_lock = asyncio.Lock()

    async def add_tick(self, tick: Dict):
        """Añadir tick al buffer"""
        async with self.tick_lock:
            self.tick_buffer.append(tick)

            if len(self.tick_buffer) >= self.batch_size:
                await self._flush_ticks()

    async def add_bar(self, bar: Dict):
        """Añadir barra al buffer"""
        async with self.bar_lock:
            self.bar_buffer.append(bar)

            if len(self.bar_buffer) >= self.batch_size:
                await self._flush_bars()

    async def add_orderbook(self, orderbook: Dict):
        """Añadir orderbook snapshot al buffer"""
        async with self.ob_lock:
            self.orderbook_buffer.append(orderbook)

            if len(self.orderbook_buffer) >= self.batch_size:
                await self._flush_orderbooks()

    async def _flush_ticks(self):
        """Flush ticks buffer to DB"""
        if not self.tick_buffer:
            return

        async with self.SessionLocal() as session:
            try:
                # INSERT ... ON CONFLICT DO NOTHING para evitar duplicados
                query = """
                    INSERT INTO market_data (time, exchange, symbol, price, quantity, is_buyer_maker, trade_id)
                    VALUES (:timestamp, :exchange, :symbol, :price, :quantity, :is_buyer_maker, :trade_id)
                    ON CONFLICT (time, exchange, symbol) DO NOTHING
                """

                await session.execute(text(query), self.tick_buffer)
                await session.commit()

                count = len(self.tick_buffer)
                self.ticks_written += count

                logger.info(
                    "ticks_flushed",
                    count=count,
                    total_written=self.ticks_written
                )

                self.tick_buffer.clear()

            except Exception as e:
                logger.error("flush_ticks_error", error=str(e))
                await session.rollback()

    async def _flush_bars(self):
        """Flush bars buffer to DB"""
        if not self.bar_buffer:
            return

        async with self.SessionLocal() as session:
            try:
                query = """
                    INSERT INTO bars (time, exchange, symbol, timeframe, open, high, low, close, volume)
                    VALUES (:timestamp, :exchange, :symbol, :timeframe, :open, :high, :low, :close, :volume)
                    ON CONFLICT (time, exchange, symbol, timeframe) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """

                await session.execute(text(query), self.bar_buffer)
                await session.commit()

                count = len(self.bar_buffer)
                self.bars_written += count

                logger.info(
                    "bars_flushed",
                    count=count,
                    total_written=self.bars_written
                )

                self.bar_buffer.clear()

            except Exception as e:
                logger.error("flush_bars_error", error=str(e))
                await session.rollback()

    async def _flush_orderbooks(self):
        """Flush orderbook snapshots buffer to DB"""
        if not self.orderbook_buffer:
            return

        async with self.SessionLocal() as session:
            try:
                # Extraer best bid/ask y calcular métricas
                processed_data = []
                for ob in self.orderbook_buffer:
                    if ob.get('bids') and ob.get('asks'):
                        best_bid = ob['bids'][0][0] if ob['bids'] else 0
                        best_ask = ob['asks'][0][0] if ob['asks'] else 0

                        # Calcular volumen top 5
                        bid_volume_5 = sum(qty for _, qty in ob['bids'][:5])
                        ask_volume_5 = sum(qty for _, qty in ob['asks'][:5])

                        processed_data.append({
                            'timestamp': ob['timestamp'],
                            'exchange': ob['exchange'],
                            'symbol': ob['symbol'],
                            'best_bid': best_bid,
                            'best_ask': best_ask,
                            'bid_volume_5': bid_volume_5,
                            'ask_volume_5': ask_volume_5,
                            'obi_5': ob.get('obi_5', 0.5),
                            'obi_10': ob.get('obi_10', 0.5),
                            'spread_pct': ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0,
                            'last_update_id': ob['last_update_id']
                        })

                if processed_data:
                    query = """
                        INSERT INTO order_book_snapshots
                        (time, exchange, symbol, best_bid, best_ask, bid_volume_5, ask_volume_5,
                         obi_5, obi_10, spread_pct, last_update_id)
                        VALUES (:timestamp, :exchange, :symbol, :best_bid, :best_ask, :bid_volume_5,
                                :ask_volume_5, :obi_5, :obi_10, :spread_pct, :last_update_id)
                        ON CONFLICT (time, exchange, symbol) DO NOTHING
                    """

                    await session.execute(text(query), processed_data)
                    await session.commit()

                    count = len(processed_data)
                    self.orderbooks_written += count

                    logger.info(
                        "orderbooks_flushed",
                        count=count,
                        total_written=self.orderbooks_written
                    )

                self.orderbook_buffer.clear()

            except Exception as e:
                logger.error("flush_orderbooks_error", error=str(e))
                await session.rollback()

    async def flush_all(self):
        """Flush todos los buffers"""
        await self._flush_ticks()
        await self._flush_bars()
        await self._flush_orderbooks()

    async def start_periodic_flush(self):
        """Iniciar flush periódico en background"""
        while True:
            await asyncio.sleep(self.flush_interval)
            try:
                await self.flush_all()
            except Exception as e:
                logger.error("periodic_flush_error", error=str(e))

    async def get_stats(self) -> Dict:
        """Obtener estadísticas del writer"""
        return {
            "ticks_written": self.ticks_written,
            "bars_written": self.bars_written,
            "orderbooks_written": self.orderbooks_written,
            "tick_buffer_size": len(self.tick_buffer),
            "bar_buffer_size": len(self.bar_buffer),
            "orderbook_buffer_size": len(self.orderbook_buffer)
        }