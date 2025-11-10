#!/usr/bin/env python3
"""
Quick Pilot Monitor - Sprint 15
Monitors real-time collection progress during 24-48h pilot
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys

def monitor_collection(base_path: str = "data/real_binance_pilot"):
    """Monitor collection progress."""
    base = Path(base_path)
    
    if not base.exists():
        print(f"❌ Path not found: {base}")
        print(f"   Make sure collector is running with --out {base_path}")
        return
    
    # Check for stats file
    stats_file = base / 'collection_stats.json'
    
    print("\n" + "="*70)
    print(f"BINANCE COLLECTOR PILOT MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for symbol in symbols:
        print(f"\n📊 {symbol}")
        print("-" * 70)
        
        # Orderbook
        ob_path = base / symbol / 'orderbook'
        if ob_path.exists():
            ob_files = list(ob_path.rglob('*.parquet'))
            if ob_files:
                try:
                    df_ob = pd.concat([pd.read_parquet(f) for f in ob_files])
                    
                    # Basic stats
                    count = len(df_ob)
                    first_ts = df_ob['timestamp'].min()
                    last_ts = df_ob['timestamp'].max()
                    duration = (last_ts - first_ts).total_seconds()
                    
                    # Coverage
                    expected = duration  # 1 Hz = 1 per second
                    coverage = (count / expected * 100) if expected > 0 else 0
                    
                    # Check for crossed markets
                    crossed = ((df_ob['bid_px_1'] >= df_ob['ask_px_1']).sum() 
                              if 'bid_px_1' in df_ob.columns and 'ask_px_1' in df_ob.columns else 0)
                    crossed_pct = (crossed / count * 100) if count > 0 else 0
                    
                    # Check gaps
                    df_ob = df_ob.sort_values('timestamp')
                    gaps = df_ob['timestamp'].diff().dt.total_seconds()
                    gaps_over_10s = (gaps > 10).sum()
                    
                    print(f"  Orderbook:")
                    print(f"    Snapshots: {count:,}")
                    print(f"    First: {first_ts}")
                    print(f"    Last:  {last_ts}")
                    print(f"    Duration: {duration/3600:.2f} hours")
                    
                    coverage_icon = "✅" if coverage > 99 else "⚠️" if coverage > 95 else "❌"
                    print(f"    Coverage: {coverage:.2f}% {coverage_icon}")
                    
                    gaps_icon = "✅" if gaps_over_10s == 0 else "⚠️" if gaps_over_10s < 5 else "❌"
                    print(f"    Gaps >10s: {gaps_over_10s} {gaps_icon}")
                    
                    crossed_icon = "✅" if crossed == 0 else "⚠️" if crossed < 10 else "❌"
                    print(f"    Crossed Markets: {crossed} ({crossed_pct:.4f}%) {crossed_icon}")
                    
                except Exception as e:
                    print(f"    ❌ Error reading data: {e}")
            else:
                print(f"    ⏳ No parquet files yet")
        else:
            print(f"    ❌ Path not found: {ob_path}")
        
        # AggTrades
        trades_path = base / symbol / 'aggtrades'
        if trades_path.exists():
            trades_files = list(trades_path.rglob('*.parquet'))
            if trades_files:
                try:
                    df_trades = pd.concat([pd.read_parquet(f) for f in trades_files[-3:]])  # Last 3 files
                    print(f"  AggTrades: {len(df_trades):,} trades")
                except Exception as e:
                    print(f"  AggTrades: Error - {e}")
            else:
                print(f"  AggTrades: ⏳ No data yet")
        else:
            print(f"  AggTrades: ❌ Path not found")
        
        # Funding
        funding_path = base / symbol / 'funding'
        if funding_path.exists():
            funding_files = list(funding_path.rglob('*.parquet'))
            if funding_files:
                try:
                    df_funding = pd.concat([pd.read_parquet(f) for f in funding_files])
                    print(f"  Funding: {len(df_funding)} updates")
                except Exception as e:
                    print(f"  Funding: Error - {e}")
            else:
                print(f"  Funding: ⏳ No data yet (normal if < 8 hours)")
        else:
            print(f"  Funding: ❌ Path not found")
    
    # Check for collection_stats.json
    if stats_file.exists():
        print("\n" + "="*70)
        print("📈 COLLECTION STATISTICS")
        print("="*70)
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"Start Time: {stats.get('start_time', 'N/A')}")
        print(f"End Time: {stats.get('end_time', 'Running...')}")
        print(f"Duration: {stats.get('duration_hours', 0):.2f} hours")
        print(f"Total Snapshots: {stats.get('total_snapshots', 0):,}")
        print(f"Total Trades: {stats.get('total_trades', 0):,}")
        print(f"Total Funding: {stats.get('total_funding', 0)}")
        print(f"Errors: {stats.get('errors', 0)}")
        print(f"Rate Limits: {stats.get('rate_limits', 0)}")
        
        print("\nPer-Symbol Stats:")
        for symbol in symbols:
            if symbol in stats.get('symbols', {}):
                s = stats['symbols'][symbol]
                print(f"\n  {symbol}:")
                print(f"    Coverage: {s.get('coverage_pct', 0):.2f}%")
                print(f"    Gaps >10s: {s.get('gaps_over_10s', 0)}")
                print(f"    Crossed Markets: {s.get('crossed_markets', 0)}")
    
    print("\n" + "="*70)
    print("PILOT CRITERIA")
    print("="*70)
    print("Target Coverage: >99%")
    print("Target Gaps: <5 total")
    print("Target Crossed: <0.1%")
    print("Target Duration: 24-48 hours")
    print("\n✅ = PASS | ⚠️ = WARNING | ❌ = FAIL")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Binance data collection pilot')
    parser.add_argument('--path', type=str, default='data/real_binance_pilot',
                       help='Base path to collection data')
    args = parser.parse_args()
    
    monitor_collection(args.path)
