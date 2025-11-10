# backend/app/backtest/data_loader.py

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd
from app.utils.logger import get_logger

logger = get_logger("backtest.data_loader")

class BinanceHistoricalDataLoader:
    """
    Descargador de datos históricos de Binance

    Uso:
        loader = BinanceHistoricalDataLoader()
        bars = await loader.download_klines(
            symbol="BTCUSDT",
            interval="1m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
    """

    BASE_URL = "https://api.binance.com"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def download_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[dict]:
        """
        Descargar klines (velas) históricas

        Args:
            symbol: Par de trading (ej: BTCUSDT)
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Fecha inicial
            end_date: Fecha final
            limit: Máximo de velas por request (max 1000)

        Returns:
            Lista de diccionarios con datos de velas
        """

        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

        logger.info(
            "downloading_klines",
            symbol=symbol,
            interval=interval,
            start_date=str(start_date),
            end_date=str(end_date)
        )

        all_bars = []
        current_start = start_date

        if not self.session:
            self.session = aiohttp.ClientSession()

        while current_start < end_date:
            # Calcular timestamps
            start_ts = int(current_start.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            # Construir URL
            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": limit
            }

            try:
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(
                            "download_failed",
                            status=response.status,
                            symbol=symbol
                        )
                        break

                    data = await response.json()

                    if not data:
                        break

                    # Parsear datos a diccionarios
                    for kline in data:
                        bar_dict = {
                            'timestamp': kline[0],
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        }
                        all_bars.append(bar_dict)

                    # Actualizar start para siguiente batch
                    last_timestamp = data[-1][0]
                    current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=1)

                    logger.info(
                        "batch_downloaded",
                        bars_count=len(data),
                        total_bars=len(all_bars),
                        last_timestamp=str(current_start)
                    )

                    # Rate limiting
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error("download_error", error=str(e), exc_info=True)
                break

        logger.info(
            "download_complete",
            symbol=symbol,
            total_bars=len(all_bars),
            start_date=str(start_date),
            end_date=str(end_date)
        )

        return all_bars

    def bars_to_dataframe(self, bars: List[dict]) -> pd.DataFrame:
        """Convertir lista de bars a DataFrame de pandas"""
        df = pd.DataFrame(bars)

        if not df.empty and 'atr' not in df.columns and {'high', 'low', 'close'}.issubset(df.columns):
            high = df['high'].to_numpy(dtype=float)
            low = df['low'].to_numpy(dtype=float)
            close = df['close'].to_numpy(dtype=float)

            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]

            true_range_components = np.vstack([
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            ])
            true_range = np.max(true_range_components, axis=0)
            atr_series = pd.Series(true_range).rolling(window=14, min_periods=1).mean()
            df['atr'] = atr_series.to_numpy()
            zero_mask = df['atr'] <= 0
            if zero_mask.any():
                df.loc[zero_mask, 'atr'] = df.loc[zero_mask, 'close'] * 0.001

        # Convertir timestamp a datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)

        return df

    def save_to_csv(
        self,
        bars: List[dict],
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Guardar bars a archivo CSV"""
        filename = f"data/{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        df = self.bars_to_dataframe(bars)
        df.to_csv(filename)

        logger.info(
            "saved_to_csv",
            filename=filename,
            rows=len(df)
        )

        return filename

    def load_from_csv(
        self,
        filename: str
    ) -> List[dict]:
        """Cargar bars desde archivo CSV"""
        df = pd.read_csv(filename, index_col='datetime', parse_dates=True)

        bars = []
        for idx, row in df.iterrows():
            bar_dict = {
                'timestamp': int(idx.timestamp() * 1000),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            bars.append(bar_dict)

        logger.info(
            "loaded_from_csv",
            filename=filename,
            bars=len(bars)
        )

        return bars