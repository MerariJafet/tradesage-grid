#!/usr/bin/env python3
"""
Diagnostic runner for backtest to check if running outside VS Code causes crash.
Runs a short backtest (1 week) with 1m bars and the SignalAggregator V2 config.
Saves a small JSON report to `reports/diagnostic_backtest.json`.
"""
import asyncio
import json
import os
import logging
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from scripts.backtest_aggregated_signals import AggregatedSignalsBacktester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run():
    config = {
        "min_confidence": 0.6,
        "volatility_filter": True,
        "session_filter": False,
        "liquidity_filter": False,
        "max_volatility_threshold": 0.05,
        "momentum_params": {},
        "mean_reversion_params": {},
        "advanced_orders_params": {
            "limit_offset": 0.001,
            "iceberg_visible": 0.05,
            "trailing_atr": 1.2,
            "stop_adaptive": True
        },
        "initial_balance": 10000.0
    }

    # Ensure local environment variables point to the host Redis when running locally
    # Docker compose uses REDIS_HOST=redis for container networking; for local runs
    # we'll prefer localhost if the redis hostname isn't resolvable.
    os.environ.setdefault('REDIS_HOST', os.environ.get('REDIS_HOST', 'redis-1'))
    os.environ.setdefault('REDIS_PORT', os.environ.get('REDIS_PORT', '6379'))

    # Ensure reports directory exists (some routines expect files under reports/)
    os.makedirs('reports', exist_ok=True)

    backtester = AggregatedSignalsBacktester(config)

    symbol = 'BTCUSDT'
    start_date = '2024-01-01'
    end_date = '2024-01-02'
    timeframe = '5m'

    try:
        report = await backtester.run_backtest(symbol, start_date, end_date, timeframe)
        os.makedirs('reports', exist_ok=True)
        out_file = f'reports/diagnostic_backtest_{start_date}_{end_date}.json'
        with open(out_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print('Diagnostic backtest completed. Report saved to', out_file)
        return 0
    except Exception as e:
        print('Diagnostic backtest failed:', str(e))
        return 2

if __name__ == '__main__':
    code = asyncio.run(run())
    exit(code)
