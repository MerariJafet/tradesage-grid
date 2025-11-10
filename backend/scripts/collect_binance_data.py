"""
Binance Futures Data Collector - Real Microstructure Data

Fetches from Binance Futures API:
- Funding Rate History (/fapi/v1/fundingRate)
- Order Book Top-5 Levels (/fapi/v1/depth?limit=5)
- Aggregate Trades (/fapi/v1/aggTrades) for CVD reconstruction

Saves daily parquet files to /data/raw/binance/{symbol}/
"""
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class BinanceDataCollector:
    """Collect real microstructure data from Binance Futures."""
    
    def __init__(self, output_dir='data/raw/binance'):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def fetch_funding_rate_history(self, symbol, start_time, end_time):
        """
        Fetch historical funding rates.
        
        Args:
            symbol: str, e.g., 'BTC/USDT'
            start_time: datetime
            end_time: datetime
            
        Returns:
            pd.DataFrame with columns: timestamp, funding_rate, mark_price
        """
        print(f"  Fetching funding rate history for {symbol}...")
        
        funding_rates = []
        current_time = start_time
        
        while current_time < end_time:
            try:
                # Binance funding rates update every 8 hours
                result = await self.exchange.fapiPublic_get_fundingrate({
                    'symbol': symbol.replace('/', ''),
                    'startTime': int(current_time.timestamp() * 1000),
                    'endTime': int(min(current_time + timedelta(days=30), end_time).timestamp() * 1000),
                    'limit': 1000
                })
                
                if not result:
                    break
                
                for item in result:
                    funding_rates.append({
                        'timestamp': pd.to_datetime(item['fundingTime'], unit='ms'),
                        'funding_rate': float(item['fundingRate']),
                        'mark_price': float(item.get('markPrice', 0))
                    })
                
                current_time += timedelta(days=30)
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                print(f"    Warning: {e}")
                break
        
        df = pd.DataFrame(funding_rates)
        if len(df) > 0:
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        print(f"    Collected {len(df)} funding rate records")
        return df
    
    async def fetch_orderbook_snapshot(self, symbol, limit=5):
        """
        Fetch current orderbook snapshot (Top-N levels).
        
        Args:
            symbol: str, e.g., 'BTC/USDT'
            limit: int, number of levels (default 5)
            
        Returns:
            dict with bid/ask prices and sizes
        """
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=limit)
            
            snapshot = {
                'timestamp': pd.to_datetime(orderbook['timestamp'], unit='ms'),
                'symbol': symbol
            }
            
            # Bids (descending price)
            for i, (price, size) in enumerate(orderbook['bids'][:limit], 1):
                snapshot[f'bid_px_{i}'] = price
                snapshot[f'bid_sz_{i}'] = size
            
            # Asks (ascending price)
            for i, (price, size) in enumerate(orderbook['asks'][:limit], 1):
                snapshot[f'ask_px_{i}'] = price
                snapshot[f'ask_sz_{i}'] = size
            
            return snapshot
            
        except Exception as e:
            print(f"    Error fetching orderbook: {e}")
            return None
    
    async def fetch_agg_trades(self, symbol, start_time, end_time):
        """
        Fetch aggregate trades for CVD reconstruction.
        
        Args:
            symbol: str, e.g., 'BTC/USDT'
            start_time: datetime
            end_time: datetime
            
        Returns:
            pd.DataFrame with columns: timestamp, price, quantity, is_buyer_maker
        """
        print(f"  Fetching aggregate trades for {symbol}...")
        
        trades = []
        current_time = start_time
        
        while current_time < end_time:
            try:
                result = await self.exchange.fapiPublic_get_aggtrades({
                    'symbol': symbol.replace('/', ''),
                    'startTime': int(current_time.timestamp() * 1000),
                    'endTime': int(min(current_time + timedelta(hours=1), end_time).timestamp() * 1000),
                    'limit': 1000
                })
                
                if not result:
                    break
                
                for trade in result:
                    trades.append({
                        'timestamp': pd.to_datetime(trade['T'], unit='ms'),
                        'price': float(trade['p']),
                        'quantity': float(trade['q']),
                        'is_buyer_maker': trade['m']  # True = sell, False = buy
                    })
                
                current_time += timedelta(hours=1)
                await asyncio.sleep(0.3)  # Rate limit
                
                if len(trades) % 10000 == 0:
                    print(f"    Progress: {len(trades):,} trades collected...")
                
            except Exception as e:
                print(f"    Warning: {e}")
                await asyncio.sleep(1)
                continue
        
        df = pd.DataFrame(trades)
        if len(df) > 0:
            df = df.drop_duplicates().sort_values('timestamp')
        
        print(f"    Collected {len(df):,} aggregate trades")
        return df
    
    async def collect_daily_data(self, symbol, date):
        """
        Collect all data for a specific day.
        
        Args:
            symbol: str, e.g., 'BTC/USDT'
            date: datetime.date
            
        Returns:
            dict with funding_rates, orderbook_snapshots, agg_trades DataFrames
        """
        print(f"\n{'='*60}")
        print(f"Collecting data for {symbol} on {date}")
        print(f"{'='*60}")
        
        start_time = datetime.combine(date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        # Fetch funding rates
        funding_df = await self.fetch_funding_rate_history(symbol, start_time, end_time)
        
        # Fetch aggregate trades
        trades_df = await self.fetch_agg_trades(symbol, start_time, end_time)
        
        # Note: Real-time orderbook snapshots require WebSocket
        # For historical analysis, we'll reconstruct from trades
        
        return {
            'funding_rates': funding_df,
            'agg_trades': trades_df,
            'date': date
        }
    
    async def save_daily_parquet(self, symbol, date, data):
        """
        Save collected data to parquet files.
        
        Args:
            symbol: str
            date: datetime.date
            data: dict with DataFrames
        """
        symbol_dir = self.output_dir / symbol.replace('/', '')
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = date.strftime('%Y-%m-%d')
        
        # Save funding rates
        if len(data['funding_rates']) > 0:
            funding_path = symbol_dir / f"funding_{date_str}.parquet"
            data['funding_rates'].to_parquet(funding_path, index=False)
            print(f"  Saved: {funding_path}")
        
        # Save aggregate trades
        if len(data['agg_trades']) > 0:
            trades_path = symbol_dir / f"trades_{date_str}.parquet"
            data['agg_trades'].to_parquet(trades_path, index=False)
            print(f"  Saved: {trades_path}")
    
    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()


async def collect_historical_range(symbols, start_date, end_date):
    """
    Collect data for multiple symbols over a date range.
    
    Args:
        symbols: list of str, e.g., ['BTC/USDT', 'ETH/USDT']
        start_date: datetime.date
        end_date: datetime.date
    """
    collector = BinanceDataCollector()
    
    try:
        current_date = start_date
        
        while current_date <= end_date:
            for symbol in symbols:
                try:
                    data = await collector.collect_daily_data(symbol, current_date)
                    await collector.save_daily_parquet(symbol, current_date, data)
                except Exception as e:
                    print(f"ERROR collecting {symbol} on {current_date}: {e}")
                    continue
            
            current_date += timedelta(days=1)
            
    finally:
        await collector.close()


if __name__ == "__main__":
    # Example: Collect last 7 days for BTC/ETH/BNB
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Binance Futures microstructure data')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
    parser.add_argument('--days', type=int, default=7, help='Number of days to collect')
    
    args = parser.parse_args()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"\n{'#'*60}")
    print(f"# Binance Data Collector - Sprint 14")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: data/raw/binance/")
    
    asyncio.run(collect_historical_range(args.symbols, start_date, end_date))
    
    print(f"\n{'='*60}")
    print("Data collection complete!")
    print(f"{'='*60}")
