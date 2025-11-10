#!/usr/bin/env python3
"""
Real Binance Futures Data Collector
Sprint 15 - Data Integrity & Real Feature Validation

Collects real-time data from Binance Futures:
- L5 Orderbook snapshots (1 Hz)
- AggTrades (continuous stream)
- Funding rates (8-hour cycle)

Storage: Parquet partitioned by symbol/date with UTC timestamps
"""

import asyncio
import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
import logging
import argparse
import json
import sqlite3
from typing import Dict, List, Optional
import time

# Configuration
BASE_URL = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
OUTPUT_DIR = Path("data/raw/binance")
CHECKPOINT_DB = Path("data/checkpoints/collector_state.db")

# Duration mapping
DURATION_MAP = {
    '48h': timedelta(hours=48),
    '24h': timedelta(hours=24),
    '90d': timedelta(days=90),
    '30d': timedelta(days=30),
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceCollector:
    """Real-time Binance Futures data collector with checkpoints."""
    
    def __init__(self, symbols: List[str], output_dir: Path, depth: int = 5, 
                 freq_seconds: float = 1.0, duration: Optional[timedelta] = None):
        self.symbols = symbols
        self.output_dir = output_dir
        self.depth = depth
        self.freq_seconds = freq_seconds
        self.duration = duration
        self.start_time = datetime.utcnow()
        self.session: Optional[aiohttp.ClientSession] = None
        self.checkpoint_db = CHECKPOINT_DB
        
        # Statistics tracking
        self.stats = {
            'start_time': self.start_time.isoformat(),
            'symbols': {},
            'total_snapshots': 0,
            'total_trades': 0,
            'total_funding': 0,
            'errors': 0,
            'rate_limits': 0
        }
        
        for symbol in symbols:
            self.stats['symbols'][symbol] = {
                'orderbook_count': 0,
                'trades_count': 0,
                'funding_count': 0,
                'gaps_over_10s': 0,
                'crossed_markets': 0,
                'last_orderbook_ts': None
            }
        
        # Create output directories
        for symbol in symbols:
            for data_type in ['orderbook', 'aggtrades', 'funding']:
                (output_dir / symbol / data_type).mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint database
        self._init_checkpoint_db()
        
        # Buffers for batch writes (reduce I/O)
        self.orderbook_buffer: Dict[str, List] = {s: [] for s in symbols}
        self.aggtrades_buffer: Dict[str, List] = {s: [] for s in symbols}
        self.buffer_size = 300  # Write every 5 minutes (300 seconds at 1 Hz)
    
    def _init_checkpoint_db(self):
        """Initialize SQLite database for checkpoints."""
        conn = sqlite3.connect(self.checkpoint_db)
        cursor = conn.cursor()
        
        # Table for aggTrade IDs (continuity tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aggtrade_checkpoint (
                symbol TEXT PRIMARY KEY,
                last_agg_id INTEGER,
                last_timestamp TEXT
            )
        ''')
        
        # Table for funding rate timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS funding_checkpoint (
                symbol TEXT PRIMARY KEY,
                last_funding_time TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Checkpoint database initialized: {self.checkpoint_db}")
    
    def _get_last_aggtrade_id(self, symbol: str) -> Optional[int]:
        """Get last collected aggTradeId for continuity."""
        conn = sqlite3.connect(self.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute('SELECT last_agg_id FROM aggtrade_checkpoint WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def _update_aggtrade_checkpoint(self, symbol: str, agg_id: int, timestamp: str):
        """Update aggTrade checkpoint."""
        conn = sqlite3.connect(self.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO aggtrade_checkpoint (symbol, last_agg_id, last_timestamp)
            VALUES (?, ?, ?)
        ''', (symbol, agg_id, timestamp))
        conn.commit()
        conn.close()
    
    def _get_last_funding_time(self, symbol: str) -> Optional[str]:
        """Get last collected funding timestamp."""
        conn = sqlite3.connect(self.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute('SELECT last_funding_time FROM funding_checkpoint WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def _update_funding_checkpoint(self, symbol: str, timestamp: str):
        """Update funding checkpoint."""
        conn = sqlite3.connect(self.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO funding_checkpoint (symbol, last_funding_time)
            VALUES (?, ?)
        ''', (symbol, timestamp))
        conn.commit()
        conn.close()
    
    async def fetch_json(self, url: str, params: dict, retries: int = 5) -> Optional[dict]:
        """Fetch JSON with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                async with self.session.get(url, params=params, timeout=10) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited. Waiting {retry_after}s")
                        self.stats['rate_limits'] += 1
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.warning(f"HTTP {response.status}: {url}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on {url} (attempt {attempt+1}/{retries})")
                self.stats['errors'] += 1
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                self.stats['errors'] += 1
            
            # Exponential backoff
            await asyncio.sleep(0.5 * (2 ** attempt))
        
        logger.error(f"Failed after {retries} retries: {url}")
        return None
    
    def write_parquet(self, df: pd.DataFrame, base_path: Path, partition_cols: List[str] = None):
        """Write DataFrame to Parquet with partitioning and compression."""
        if df.empty:
            return
        
        try:
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            if partition_cols:
                pq.write_to_dataset(
                    table,
                    root_path=str(base_path),
                    partition_cols=partition_cols,
                    compression='snappy',
                    use_dictionary=True,
                    compression_level=None
                )
            else:
                # Non-partitioned write (for small files)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                file_path = base_path / f"{timestamp}.parquet"
                pq.write_table(table, file_path, compression='snappy')
            
            logger.debug(f"Written {len(df)} records to {base_path}")
        except Exception as e:
            logger.error(f"Error writing parquet to {base_path}: {e}")
    
    async def collect_orderbook(self, symbol: str):
        """Collect L5 orderbook snapshots at specified frequency."""
        url = f"{BASE_URL}/fapi/v1/depth"
        logger.info(f"Starting orderbook collection for {symbol} at {self.freq_seconds}s frequency")
        
        last_timestamp = None
        
        while True:
            # Check duration limit
            if self.duration and (datetime.utcnow() - self.start_time) >= self.duration:
                logger.info(f"Duration limit reached for {symbol} orderbook collection")
                break
            
            try:
                data = await self.fetch_json(url, {"symbol": symbol, "limit": self.depth})
                
                if data and 'bids' in data and 'asks' in data:
                    ts = pd.Timestamp.utcnow().tz_localize(None)  # Remove timezone info for consistency
                    bids = data['bids'][:self.depth]
                    asks = data['asks'][:self.depth]
                    
                    # Check for gap
                    if last_timestamp:
                        gap = (ts - last_timestamp).total_seconds()
                        if gap > 10:
                            self.stats['symbols'][symbol]['gaps_over_10s'] += 1
                            logger.warning(f"{symbol}: Gap detected: {gap:.1f}s")
                    
                    last_timestamp = ts
                    
                    # Check for crossed market
                    if len(bids) > 0 and len(asks) > 0:
                        bid_px = float(bids[0][0])
                        ask_px = float(asks[0][0])
                        if bid_px >= ask_px:
                            self.stats['symbols'][symbol]['crossed_markets'] += 1
                            logger.error(f"{symbol}: CROSSED MARKET - bid={bid_px} >= ask={ask_px}")
                    
                    # Flatten to wide format (one row per snapshot)
                    row = {
                        'timestamp': ts,
                        'symbol': symbol,
                        'date': ts.strftime('%Y-%m-%d'),
                    }
                    
                    # Add bid levels
                    for i in range(self.depth):
                        if i < len(bids):
                            row[f'bid_px_{i+1}'] = float(bids[i][0])
                            row[f'bid_sz_{i+1}'] = float(bids[i][1])
                        else:
                            row[f'bid_px_{i+1}'] = None
                            row[f'bid_sz_{i+1}'] = None
                    
                    # Add ask levels
                    for i in range(self.depth):
                        if i < len(asks):
                            row[f'ask_px_{i+1}'] = float(asks[i][0])
                            row[f'ask_sz_{i+1}'] = float(asks[i][1])
                        else:
                            row[f'ask_px_{i+1}'] = None
                            row[f'ask_sz_{i+1}'] = None
                    
                    # Add to buffer
                    self.orderbook_buffer[symbol].append(row)
                    self.stats['symbols'][symbol]['orderbook_count'] += 1
                    self.stats['symbols'][symbol]['last_orderbook_ts'] = ts.isoformat()
                    self.stats['total_snapshots'] += 1
                    
                    # Flush buffer if full
                    if len(self.orderbook_buffer[symbol]) >= self.buffer_size:
                        df = pd.DataFrame(self.orderbook_buffer[symbol])
                        base_path = self.output_dir / symbol / 'orderbook'
                        self.write_parquet(df, base_path, partition_cols=['date'])
                        self.orderbook_buffer[symbol] = []
                        logger.info(f"Flushed {len(df)} orderbook snapshots for {symbol}")
                
                await asyncio.sleep(self.freq_seconds)
                
            except Exception as e:
                logger.error(f"Error in orderbook collection for {symbol}: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5.0)
    
    async def collect_aggtrades(self, symbol: str):
        """Collect aggregate trades with continuity tracking."""
        url = f"{BASE_URL}/fapi/v1/aggTrades"
        logger.info(f"Starting aggTrades collection for {symbol}")
        
        # Get last aggTradeId from checkpoint
        from_id = self._get_last_aggtrade_id(symbol)
        
        while True:
            # Check duration limit
            if self.duration and (datetime.utcnow() - self.start_time) >= self.duration:
                logger.info(f"Duration limit reached for {symbol} aggTrades collection")
                break
            
            try:
                params = {"symbol": symbol, "limit": 1000}
                if from_id:
                    params['fromId'] = from_id + 1
                
                data = await self.fetch_json(url, params)
                
                if data and len(data) > 0:
                    rows = []
                    for trade in data:
                        ts = pd.Timestamp(trade['T'], unit='ms', tz='UTC').tz_convert(None)  # Remove timezone
                        rows.append({
                            'timestamp': ts,
                            'symbol': symbol,
                            'agg_trade_id': trade['a'],
                            'price': float(trade['p']),
                            'quantity': float(trade['q']),
                            'first_trade_id': trade['f'],
                            'last_trade_id': trade['l'],
                            'is_buyer_maker': trade['m'],  # True = sell, False = buy
                            'date': ts.strftime('%Y-%m-%d')
                        })
                    
                    # Update checkpoint with last ID
                    last_agg_id = data[-1]['a']
                    last_ts = pd.Timestamp(data[-1]['T'], unit='ms', tz='UTC').tz_convert(None).isoformat()
                    self._update_aggtrade_checkpoint(symbol, last_agg_id, last_ts)
                    from_id = last_agg_id
                    
                    # Add to buffer
                    self.aggtrades_buffer[symbol].extend(rows)
                    self.stats['symbols'][symbol]['trades_count'] += len(rows)
                    self.stats['total_trades'] += len(rows)
                    
                    # Flush buffer if full
                    if len(self.aggtrades_buffer[symbol]) >= self.buffer_size * 10:  # Larger buffer for trades
                        df = pd.DataFrame(self.aggtrades_buffer[symbol])
                        base_path = self.output_dir / symbol / 'aggtrades'
                        self.write_parquet(df, base_path, partition_cols=['date'])
                        self.aggtrades_buffer[symbol] = []
                        logger.info(f"Flushed {len(df)} aggTrades for {symbol}")
                    
                    logger.debug(f"Collected {len(data)} aggTrades for {symbol} (last ID: {last_agg_id})")
                
                await asyncio.sleep(1.0)  # Poll every second
                
            except Exception as e:
                logger.error(f"Error in aggTrades collection for {symbol}: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5.0)
    
    async def collect_funding(self, symbol: str):
        """Collect funding rates (8-hour updates)."""
        url = f"{BASE_URL}/fapi/v1/fundingRate"
        logger.info(f"Starting funding rate collection for {symbol}")
        
        while True:
            # Check duration limit
            if self.duration and (datetime.utcnow() - self.start_time) >= self.duration:
                logger.info(f"Duration limit reached for {symbol} funding collection")
                break
            
            try:
                # Get last collected funding time
                last_time_str = self._get_last_funding_time(symbol)
                
                if last_time_str:
                    last_time = pd.Timestamp(last_time_str)
                    # Remove timezone if present (normalize to naive UTC)
                    if last_time.tz is not None:
                        last_time = last_time.tz_convert(None)
                    start_time = int((last_time + timedelta(hours=8)).timestamp() * 1000)
                else:
                    # Start from 90 days ago
                    start_time = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)
                
                end_time = int(datetime.utcnow().timestamp() * 1000)
                
                params = {
                    "symbol": symbol,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000
                }
                
                data = await self.fetch_json(url, params)
                
                if data and len(data) > 0:
                    rows = []
                    for funding in data:
                        ts = pd.Timestamp(funding['fundingTime'], unit='ms', tz='UTC').tz_convert(None)  # Remove timezone
                        rows.append({
                            'timestamp': ts,
                            'symbol': symbol,
                            'funding_rate': float(funding['fundingRate']),
                            'mark_price': float(funding.get('markPrice', 0)),
                            'date': ts.strftime('%Y-%m-%d')
                        })
                    
                    # Write immediately (funding is sparse)
                    df = pd.DataFrame(rows)
                    base_path = self.output_dir / symbol / 'funding'
                    self.write_parquet(df, base_path, partition_cols=['date'])
                    
                    # Update checkpoint
                    last_ts = pd.Timestamp(data[-1]['fundingTime'], unit='ms', tz='UTC').tz_convert(None).isoformat()
                    self._update_funding_checkpoint(symbol, last_ts)
                    
                    self.stats['symbols'][symbol]['funding_count'] += len(rows)
                    self.stats['total_funding'] += len(rows)
                    
                    logger.info(f"Collected {len(df)} funding rate updates for {symbol}")
                
                # Check every hour, but only new 8-hour updates will be returned
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in funding collection for {symbol}: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(300)  # 5 minutes
    
    async def flush_all_buffers(self):
        """Periodic flush of all buffers (safety mechanism)."""
        while True:
            # Check duration limit
            if self.duration and (datetime.utcnow() - self.start_time) >= self.duration:
                logger.info("Duration limit reached for periodic flush")
                break
            
            await asyncio.sleep(600)  # Every 10 minutes
            
            for symbol in self.symbols:
                # Flush orderbook buffer
                if self.orderbook_buffer[symbol]:
                    df = pd.DataFrame(self.orderbook_buffer[symbol])
                    base_path = self.output_dir / symbol / 'orderbook'
                    self.write_parquet(df, base_path, partition_cols=['date'])
                    logger.info(f"Periodic flush: {len(df)} orderbook snapshots for {symbol}")
                    self.orderbook_buffer[symbol] = []
                
                # Flush aggtrades buffer
                if self.aggtrades_buffer[symbol]:
                    df = pd.DataFrame(self.aggtrades_buffer[symbol])
                    base_path = self.output_dir / symbol / 'aggtrades'
                    self.write_parquet(df, base_path, partition_cols=['date'])
                    logger.info(f"Periodic flush: {len(df)} aggTrades for {symbol}")
                    self.aggtrades_buffer[symbol] = []
    
    def save_stats(self):
        """Save collection statistics to JSON."""
        self.stats['end_time'] = datetime.utcnow().isoformat()
        duration = datetime.utcnow() - self.start_time
        self.stats['duration_seconds'] = duration.total_seconds()
        self.stats['duration_hours'] = duration.total_seconds() / 3600
        
        # Calculate coverage for each symbol
        for symbol in self.symbols:
            count = self.stats['symbols'][symbol]['orderbook_count']
            if count > 0:
                expected = duration.total_seconds() / self.freq_seconds
                coverage = (count / expected) * 100
                self.stats['symbols'][symbol]['coverage_pct'] = round(coverage, 2)
        
        # Save to file
        stats_path = self.output_dir / 'collection_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COLLECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Duration: {self.stats['duration_hours']:.2f} hours")
        logger.info(f"Total Snapshots: {self.stats['total_snapshots']:,}")
        logger.info(f"Total Trades: {self.stats['total_trades']:,}")
        logger.info(f"Total Funding: {self.stats['total_funding']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Rate Limits: {self.stats['rate_limits']}")
        
        for symbol in self.symbols:
            s = self.stats['symbols'][symbol]
            logger.info(f"\n{symbol}:")
            logger.info(f"  Orderbook: {s['orderbook_count']:,} snapshots")
            logger.info(f"  Coverage: {s.get('coverage_pct', 0):.2f}%")
            logger.info(f"  Gaps >10s: {s['gaps_over_10s']}")
            logger.info(f"  Crossed Markets: {s['crossed_markets']}")
            logger.info(f"  Trades: {s['trades_count']:,}")
            logger.info(f"  Funding: {s['funding_count']}")
        
        logger.info("="*60)
    
    async def run(self):
        """Main collection loop."""
        logger.info(f"Starting Binance Futures data collector for {self.symbols}")
        logger.info(f"Depth: L{self.depth}, Frequency: {self.freq_seconds}s, Duration: {self.duration or 'unlimited'}")
        
        # Create session
        self.session = aiohttp.ClientSession()
        
        try:
            # Create tasks for all symbols and data types
            tasks = []
            
            for symbol in self.symbols:
                tasks.append(self.collect_orderbook(symbol))
                tasks.append(self.collect_aggtrades(symbol))
                tasks.append(self.collect_funding(symbol))
            
            # Add periodic flush task
            tasks.append(self.flush_all_buffers())
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal, flushing buffers...")
            
        finally:
            # Final flush of all buffers
            for symbol in self.symbols:
                if self.orderbook_buffer[symbol]:
                    df = pd.DataFrame(self.orderbook_buffer[symbol])
                    base_path = self.output_dir / symbol / 'orderbook'
                    self.write_parquet(df, base_path, partition_cols=['date'])
                    logger.info(f"Final flush: {len(df)} orderbook for {symbol}")
                
                if self.aggtrades_buffer[symbol]:
                    df = pd.DataFrame(self.aggtrades_buffer[symbol])
                    base_path = self.output_dir / symbol / 'aggtrades'
                    self.write_parquet(df, base_path, partition_cols=['date'])
                    logger.info(f"Final flush: {len(df)} aggTrades for {symbol}")
            
            await self.session.close()
            
            # Save statistics
            self.save_stats()
            
            logger.info("Collector stopped cleanly")


def main():
    parser = argparse.ArgumentParser(description='Real Binance Futures Data Collector')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to collect')
    parser.add_argument('--out', type=str, default=str(OUTPUT_DIR), help='Output directory')
    parser.add_argument('--depth', type=int, default=5, help='Orderbook depth (default: 5)')
    parser.add_argument('--freq', type=str, default='1s', help='Collection frequency (e.g., 1s, 2s)')
    parser.add_argument('--duration', type=str, default=None, 
                       help='Collection duration (e.g., 48h, 24h, 90d, None for unlimited)')
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse frequency (e.g., "1s" -> 1.0)
    freq_str = args.freq.lower()
    if freq_str.endswith('s'):
        freq_seconds = float(freq_str[:-1])
    else:
        freq_seconds = float(freq_str)
    
    # Parse duration
    duration = None
    if args.duration:
        if args.duration in DURATION_MAP:
            duration = DURATION_MAP[args.duration]
        else:
            # Try parsing custom format (e.g., "7d", "12h")
            duration_str = args.duration.lower()
            if duration_str.endswith('d'):
                duration = timedelta(days=int(duration_str[:-1]))
            elif duration_str.endswith('h'):
                duration = timedelta(hours=int(duration_str[:-1]))
            else:
                logger.error(f"Invalid duration format: {args.duration}")
                return
    
    logger.info(f"Configuration:")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Depth: L{args.depth}")
    logger.info(f"  Frequency: {freq_seconds}s")
    logger.info(f"  Duration: {duration or 'unlimited'}")
    
    # Create and run collector
    collector = BinanceCollector(args.symbols, output_dir, args.depth, freq_seconds, duration)
    
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        logger.info("Collector stopped by user")


if __name__ == "__main__":
    main()
