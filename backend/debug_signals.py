"""
Debug script to check indicator values and signal generation
"""

import sys
import importlib
import asyncio
import pandas as pd
from datetime import datetime

# Force reload of modules to avoid caching issues
if 'app.core.strategies.momentum_scalping' in sys.modules:
    importlib.reload(sys.modules['app.core.strategies.momentum_scalping'])
if 'app.core.strategies.position_sizer' in sys.modules:
    importlib.reload(sys.modules['app.core.strategies.position_sizer'])

from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.momentum_scalping import MomentumScalpingStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.backtest.data_loader import BinanceHistoricalDataLoader

async def debug_signals():
    """Debug signal generation process"""

    # Load some test data
    data_loader = BinanceHistoricalDataLoader()
    async with data_loader as loader:
        bars = await loader.download_klines(
            symbol="BTCUSDT",
            interval="5m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2)  # Just one day for testing
        )

    if not bars:
        print("No data available")
        return

    data = data_loader.bars_to_dataframe(bars)
    data.set_index('timestamp', inplace=True)

    print(f"Loaded {len(data)} bars")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")

    # Initialize components
    indicator_manager = IndicatorManager()
    indicator_manager.initialize_symbol("BTCUSDT")
    position_sizer = PositionSizer(account_balance=10000.0)

    # Create strategy with a mock signal validator that always validates
    class MockSignalValidator:
        async def validate(self, signal, market_data=None):
            return True, []  # Always valid for debugging

    signal_validator = MockSignalValidator()

    # Debug: Check PositionSizer methods
    print("PositionSizer methods:", [m for m in dir(position_sizer) if not m.startswith('_')])
    print("Has calculate_quantity:", hasattr(position_sizer, 'calculate_quantity'))
    print("Has calculate_position_size:", hasattr(position_sizer, 'calculate_position_size'))
    print("Feeding historical data to indicators...")
    for idx, (_, row) in enumerate(data.iterrows()):
        bar_dict = {
            'timestamp': row.name,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        }
        indicator_manager.update_with_bar("BTCUSDT", bar_dict)

        if idx % 100 == 0:
            print(f"Processed {idx} bars...")

    # Create strategy
    strategy = MomentumScalpingStrategy(
        symbol="BTCUSDT",
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator
    )

    print(f"Strategy min_periods: {strategy.min_periods}")

    # Test signal generation on last 10 bars
    print("\nTesting signal generation on last 10 bars:")
    signals_generated = 0

    for i in range(max(strategy.min_periods, len(data) - 10), len(data)):
        row = data.iloc[i]
        bar_dict = {
            'timestamp': row.name,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        }

        # Get indicator values
        indicators = indicator_manager.get_all_values("BTCUSDT")

        # Check key indicators
        macd_line = indicators.get('macd_line')
        macd_signal = indicators.get('macd_signal')
        rsi = indicators.get('rsi_14')
        current_volume = bar_dict['volume']

        print(f"\nBar {i}: {bar_dict['timestamp']}")
        print(f"  Close: {bar_dict['close']:.2f}")
        print(f"  Volume: {current_volume}")
        print(f"  MACD Line: {macd_line}")
        print(f"  MACD Signal: {macd_signal}")
        print(f"  RSI: {rsi}")

        # Check conditions
        volume_ok = current_volume > 1
        long_conditions = (
            macd_line is not None and macd_signal is not None and rsi is not None and
            macd_line > macd_signal and
            rsi > 45 and
            volume_ok
        )

        short_conditions = (
            macd_line is not None and macd_signal is not None and rsi is not None and
            macd_line < macd_signal and
            rsi < 55 and
            volume_ok
        )

        print(f"  Volume OK: {volume_ok}")
        print(f"  Long conditions: {long_conditions}")
        print(f"  Short conditions: {short_conditions}")

        # Try to generate signal
        market_data = {
            'bar': bar_dict,
            'indicators': {}
        }

        signal = await strategy.generate_signal(market_data)
        if signal:
            signals_generated += 1
            print(f"  SIGNAL GENERATED: {signal.action} at {signal.entry_price}")
        else:
            print("  No signal")

    print(f"\nTotal signals generated: {signals_generated}")

if __name__ == "__main__":
    asyncio.run(debug_signals())