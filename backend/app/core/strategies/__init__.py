# Strategy Engine Package
# Sprint 4: Strategy Engine - Breakout de Compresión

from .signal import TradingSignal, SignalAction, SignalType
from .position import Position, PositionSide, PositionStatus
from .position_sizer import PositionSizer
from .signal_validator import SignalValidator
from .base import BaseStrategy
from .breakout_compression import BreakoutCompressionStrategy
from .strategy_manager import StrategyManager

__all__ = [
    'TradingSignal',
    'SignalAction',
    'SignalType',
    'Position',
    'PositionSide',
    'PositionStatus',
    'PositionSizer',
    'SignalValidator',
    'BaseStrategy',
    'BreakoutCompressionStrategy',
    'StrategyManager'
]