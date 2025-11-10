#!/usr/bin/env python3
"""
Binance Futures WebSocket Data Collector
Collects L5 orderbook (@depth5@100ms), aggTrades, and mark price/funding via WebSocket streams.
Target: >99% coverage, <5 gaps >10s, <200ms latency.

Usage:
    python real_binance_socket_collector.py \\
        --symbols BTCUSDT ETHUSDT BNBUSDT \\
        --duration 12h \\
        --out data/real_binance_ws_pilot

Author: Abinadab (AI Assistant)
Date: 2025-11-02
Sprint: 15 (WebSocket Migration)
"""

import asyncio
import json
import logging
import sqlite3
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/socket_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceWebSocketCollector:
    """
    WebSocket-based Binance Futures data collector.
    Streams: @depth5@100ms, @aggTrade, @markPrice@1s
    """
    
    # WebSocket endpoints
    WS_BASE = "wss://fstream.binance.com"
    
    # Buffer sizes
    ORDERBOOK_BUFFER_SIZE = 6000   # 100ms streams → flush every 10 minutes
    TRADES_BUFFER_SIZE = 50000     # High-frequency trades
    MARKPRICE_BUFFER_SIZE = 600    # 1s updates → flush every 10 minutes
    
    # Duration mappings
    DURATION_MAP = {
        '12h': timedelta(hours=12),
        '24h': timedelta(hours=24),
        '48h': timedelta(hours=48),
        '7d': timedelta(days=7),
        '30d': timedelta(days=30),
        '90d': timedelta(days=90),
    }
    
    def __init__(
        self, 
        symbols: List[str], 
        output_dir: Path,
        duration: Optional[timedelta] = None
    ):
        self.symbols = [s.lower() for s in symbols]  # WebSocket uses lowercase
        self.output_dir = Path(output_dir)
        self.duration = duration
        self.start_time = datetime.utcnow()
        
        # Create output directories
        for symbol in symbols:
            for data_type in ['orderbook', 'aggtrades', 'markprice']:
                (self.output_dir / symbol.upper() / data_type).mkdir(parents=True, exist_ok=True)
        
        # Buffers for each data type
        self.orderbook_buffer = defaultdict(list)
        self.trades_buffer = defaultdict(list)
        self.markprice_buffer = defaultdict(list)
        
        # Statistics tracking
        self.stats = {
            'start_time': self.start_time.isoformat(),
            'symbols': {
                s.upper(): {
                    'orderbook_count': 0,
                    'trades_count': 0,
                    'markprice_count': 0,
                    'last_orderbook_ts': None,
                    'last_trade_ts': None,
                    'last_markprice_ts': None,
                    'gaps_over_10s': 0,
                    'crossed_markets': 0,
                    'websocket_reconnects': 0,
                }
                for s in symbols
            },
            'total_snapshots': 0,
            'total_trades': 0,
            'total_markprice': 0,
            'websocket_errors': 0,
        }
        
        # Last timestamps for gap detection
        self.last_orderbook_ts = {s.upper(): None for s in symbols}
        
        # Shutdown flag
        self.shutdown = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown = True
    
    def _build_stream_url(self) -> str:
        """Build combined WebSocket stream URL for all symbols."""
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol}@depth5@100ms")  # L5 orderbook every 100ms
            streams.append(f"{symbol}@aggTrade")      # Aggregate trades
            streams.append(f"{symbol}@markPrice@1s")  # Mark price + funding (1s)
        
        # Combined stream endpoint
        stream_names = '/'.join(streams)
        url = f"{self.WS_BASE}/stream?streams={stream_names}"
        return url
    
    async def _handle_depth(self, data: dict):
        """Handle @depth5@100ms messages."""
        symbol = data['s']
        ts = pd.Timestamp(data['T'], unit='ms')
        
        bids = data['b']  # [[price, qty], ...]
        asks = data['a']
        
        # Gap detection
        if self.last_orderbook_ts[symbol]:
            gap = (ts - self.last_orderbook_ts[symbol]).total_seconds()
            if gap > 10:
                self.stats['symbols'][symbol]['gaps_over_10s'] += 1
                logger.warning(f"{symbol}: Gap detected in orderbook: {gap:.1f}s")
        
        self.last_orderbook_ts[symbol] = ts
        
        # Crossed market detection
        if len(bids) > 0 and len(asks) > 0:
            bid_px = float(bids[0][0])
            ask_px = float(asks[0][0])
            if bid_px >= ask_px:
                self.stats['symbols'][symbol]['crossed_markets'] += 1
                logger.error(f"{symbol}: CROSSED MARKET - bid={bid_px} >= ask={ask_px}")
        
        # Build row (same format as REST collector)
        row = {
            'timestamp': ts,
            'symbol': symbol,
            'date': ts.strftime('%Y-%m-%d'),
        }
        
        # Add bid levels
        for i in range(5):
            if i < len(bids):
                row[f'bid_px_{i+1}'] = float(bids[i][0])
                row[f'bid_sz_{i+1}'] = float(bids[i][1])
            else:
                row[f'bid_px_{i+1}'] = None
                row[f'bid_sz_{i+1}'] = None
        
        # Add ask levels
        for i in range(5):
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
        
        # Flush if buffer full
        if len(self.orderbook_buffer[symbol]) >= self.ORDERBOOK_BUFFER_SIZE:
            await self._flush_orderbook(symbol)
    
    async def _handle_aggtrade(self, data: dict):
        """Handle @aggTrade messages."""
        symbol = data['s']
        ts = pd.Timestamp(data['T'], unit='ms')
        
        row = {
            'timestamp': ts,
            'symbol': symbol,
            'agg_trade_id': data['a'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'first_trade_id': data['f'],
            'last_trade_id': data['l'],
            'is_buyer_maker': data['m'],  # True = sell, False = buy
            'date': ts.strftime('%Y-%m-%d')
        }
        
        self.trades_buffer[symbol].append(row)
        self.stats['symbols'][symbol]['trades_count'] += 1
        self.stats['symbols'][symbol]['last_trade_ts'] = ts.isoformat()
        self.stats['total_trades'] += 1
        
        # Flush if buffer full
        if len(self.trades_buffer[symbol]) >= self.TRADES_BUFFER_SIZE:
            await self._flush_trades(symbol)
    
    async def _handle_markprice(self, data: dict):
        """Handle @markPrice@1s messages."""
        symbol = data['s']
        ts = pd.Timestamp(data['E'], unit='ms')  # Event time
        
        row = {
            'timestamp': ts,
            'symbol': symbol,
            'mark_price': float(data['p']),
            'index_price': float(data['i']),
            'estimated_settle_price': float(data['P']),
            'funding_rate': float(data['r']),
            'next_funding_time': pd.Timestamp(data['T'], unit='ms'),
            'date': ts.strftime('%Y-%m-%d')
        }
        
        self.markprice_buffer[symbol].append(row)
        self.stats['symbols'][symbol]['markprice_count'] += 1
        self.stats['symbols'][symbol]['last_markprice_ts'] = ts.isoformat()
        self.stats['total_markprice'] += 1
        
        # Flush if buffer full
        if len(self.markprice_buffer[symbol]) >= self.MARKPRICE_BUFFER_SIZE:
            await self._flush_markprice(symbol)
    
    async def _flush_orderbook(self, symbol: str):
        """Flush orderbook buffer to parquet."""
        if not self.orderbook_buffer[symbol]:
            return
        
        df = pd.DataFrame(self.orderbook_buffer[symbol])
        base_path = self.output_dir / symbol / 'orderbook'
        self._write_parquet(df, base_path, partition_cols=['date'])
        count = len(df)
        self.orderbook_buffer[symbol] = []
        logger.info(f"Flushed {count} orderbook snapshots for {symbol}")
    
    async def _flush_trades(self, symbol: str):
        """Flush trades buffer to parquet."""
        if not self.trades_buffer[symbol]:
            return
        
        df = pd.DataFrame(self.trades_buffer[symbol])
        base_path = self.output_dir / symbol / 'aggtrades'
        self._write_parquet(df, base_path, partition_cols=['date'])
        count = len(df)
        self.trades_buffer[symbol] = []
        logger.info(f"Flushed {count} aggTrades for {symbol}")
    
    async def _flush_markprice(self, symbol: str):
        """Flush mark price buffer to parquet."""
        if not self.markprice_buffer[symbol]:
            return
        
        df = pd.DataFrame(self.markprice_buffer[symbol])
        base_path = self.output_dir / symbol / 'markprice'
        self._write_parquet(df, base_path, partition_cols=['date'])
        count = len(df)
        self.markprice_buffer[symbol] = []
        logger.info(f"Flushed {count} markPrice updates for {symbol}")
    
    def _write_parquet(self, df: pd.DataFrame, base_path: Path, partition_cols: List[str]):
        """Write DataFrame to partitioned parquet files."""
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Check if directory has existing parquet files
        has_existing_files = any(base_path.glob('**/*.parquet'))
        
        df.to_parquet(
            base_path,
            engine='fastparquet',
            compression='snappy',
            partition_on=partition_cols,
            append=has_existing_files  # Only append if files exist
        )
    
    async def _flush_all_buffers(self):
        """Flush all buffers (called periodically and on shutdown)."""
        for symbol in [s.upper() for s in self.symbols]:
            await self._flush_orderbook(symbol)
            await self._flush_trades(symbol)
            await self._flush_markprice(symbol)
    
    async def _periodic_flush(self):
        """Periodic buffer flush every 5 minutes."""
        while not self.shutdown:
            await asyncio.sleep(300)  # 5 minutes
            
            # Check duration limit
            if self.duration and (datetime.utcnow() - self.start_time) >= self.duration:
                logger.info("Duration limit reached, initiating shutdown...")
                self.shutdown = True
                break
            
            logger.info("Periodic buffer flush...")
            await self._flush_all_buffers()
    
    async def _handle_message(self, message: dict):
        """Route WebSocket message to appropriate handler."""
        try:
            stream = message['stream']
            data = message['data']
            
            if '@depth5@100ms' in stream:
                await self._handle_depth(data)
            elif '@aggTrade' in stream:
                await self._handle_aggtrade(data)
            elif '@markPrice@1s' in stream:
                await self._handle_markprice(data)
            else:
                logger.warning(f"Unknown stream: {stream}")
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.stats['websocket_errors'] += 1
    
    async def _connect_and_stream(self):
        """Main WebSocket connection loop with auto-reconnect."""
        url = self._build_stream_url()
        reconnect_delay = 1  # Start with 1 second
        max_reconnect_delay = 60  # Max 60 seconds
        
        while not self.shutdown:
            try:
                logger.info(f"Connecting to WebSocket: {url[:100]}...")
                
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("WebSocket connected successfully")
                    reconnect_delay = 1  # Reset delay on successful connection
                    
                    # Increment reconnect counter for all symbols
                    for symbol in [s.upper() for s in self.symbols]:
                        self.stats['symbols'][symbol]['websocket_reconnects'] += 1
                    
                    # Read messages
                    async for message in ws:
                        if self.shutdown:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
            
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                if not self.shutdown:
                    logger.info(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if not self.shutdown:
                    logger.info(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        
        logger.info("WebSocket connection loop exited")
    
    def _save_stats(self):
        """Save collection statistics to JSON."""
        self.stats['end_time'] = datetime.utcnow().isoformat()
        duration = datetime.utcnow() - self.start_time
        self.stats['duration_hours'] = duration.total_seconds() / 3600
        
        # Calculate coverage for each symbol
        for symbol in [s.upper() for s in self.symbols]:
            count = self.stats['symbols'][symbol]['orderbook_count']
            if self.duration:
                expected = (self.duration.total_seconds() / 0.1)  # 100ms = 0.1s
                coverage = (count / expected) * 100 if expected > 0 else 0
                self.stats['symbols'][symbol]['coverage_pct'] = round(coverage, 2)
            else:
                self.stats['symbols'][symbol]['coverage_pct'] = None
        
        # Save to file
        stats_file = self.output_dir / 'collection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
        
        # Print summary
        logger.info("=" * 70)
        logger.info("COLLECTION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Start Time: {self.stats['start_time']}")
        logger.info(f"End Time: {self.stats['end_time']}")
        logger.info(f"Duration: {self.stats['duration_hours']:.2f} hours")
        logger.info(f"Total Snapshots: {self.stats['total_snapshots']:,}")
        logger.info(f"Total Trades: {self.stats['total_trades']:,}")
        logger.info(f"Total MarkPrice: {self.stats['total_markprice']:,}")
        logger.info(f"WebSocket Errors: {self.stats['websocket_errors']}")
        
        for symbol in [s.upper() for s in self.symbols]:
            stats = self.stats['symbols'][symbol]
            logger.info(f"\n{symbol}:")
            logger.info(f"  Orderbook: {stats['orderbook_count']:,} snapshots")
            if stats['coverage_pct'] is not None:
                logger.info(f"  Coverage: {stats['coverage_pct']:.2f}%")
            logger.info(f"  Gaps >10s: {stats['gaps_over_10s']}")
            logger.info(f"  Crossed Markets: {stats['crossed_markets']}")
            logger.info(f"  Trades: {stats['trades_count']:,}")
            logger.info(f"  MarkPrice: {stats['markprice_count']:,}")
            logger.info(f"  Reconnects: {stats['websocket_reconnects']}")
        
        logger.info("=" * 70)
    
    async def run(self):
        """Main execution loop."""
        logger.info(f"Starting WebSocket collector for {[s.upper() for s in self.symbols]}")
        if self.duration:
            logger.info(f"Duration: {self.duration}")
        logger.info(f"Output: {self.output_dir}")
        
        # Start tasks
        tasks = [
            asyncio.create_task(self._connect_and_stream()),
            asyncio.create_task(self._periodic_flush()),
        ]
        
        # Wait for completion or shutdown
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final flush
        logger.info("Flushing final buffers...")
        await self._flush_all_buffers()
        
        # Save statistics
        self._save_stats()
        
        logger.info("Collector stopped cleanly")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance Futures WebSocket Data Collector')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to collect (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--duration', type=str, help='Collection duration (e.g., 12h, 24h, 7d, 90d)')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Parse duration
    duration = None
    if args.duration:
        if args.duration in BinanceWebSocketCollector.DURATION_MAP:
            duration = BinanceWebSocketCollector.DURATION_MAP[args.duration]
        else:
            # Try parsing custom format like "6h", "3d"
            try:
                if args.duration.endswith('h'):
                    duration = timedelta(hours=int(args.duration[:-1]))
                elif args.duration.endswith('d'):
                    duration = timedelta(days=int(args.duration[:-1]))
                else:
                    logger.error(f"Invalid duration format: {args.duration}")
                    sys.exit(1)
            except ValueError:
                logger.error(f"Invalid duration: {args.duration}")
                sys.exit(1)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Output: {args.out}")
    logger.info(f"  Duration: {duration if duration else 'unlimited'}")
    
    # Create collector and run
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collector = BinanceWebSocketCollector(
        symbols=args.symbols,
        output_dir=output_dir,
        duration=duration
    )
    
    # Run async main loop
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        logger.info("Collector stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
