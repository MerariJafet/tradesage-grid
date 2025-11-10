#!/usr/bin/env python3
"""
Simple test to check if the strategy has the correct method
"""

import sys
sys.path.insert(0, '/Users/merari/Desktop/bot de scalping/backend')

from app.core.strategies.momentum_scalping import MomentumScalpingStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.core.indicators.indicator_manager import IndicatorManager

# Create instances
position_sizer = PositionSizer(account_balance=10000.0)
indicator_manager = IndicatorManager()
signal_validator = SignalValidator(indicator_manager=indicator_manager)

# Check if the method exists
print("Available methods in PositionSizer:")
methods = [method for method in dir(position_sizer) if not method.startswith('_')]
for method in methods:
    print(f"  - {method}")

# Check if calculate_quantity exists
if hasattr(position_sizer, 'calculate_quantity'):
    print("✓ calculate_quantity method exists")
else:
    print("✗ calculate_quantity method does NOT exist")

# Check if calculate_position_size exists
if hasattr(position_sizer, 'calculate_position_size'):
    print("✗ calculate_position_size method exists (should not)")
else:
    print("✓ calculate_position_size method does not exist (correct)")

# Try to create strategy
try:
    strategy = MomentumScalpingStrategy(
        symbol="BTCUSDT",
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator
    )
    print("✓ Strategy created successfully")
except Exception as e:
    print(f"✗ Error creating strategy: {e}")