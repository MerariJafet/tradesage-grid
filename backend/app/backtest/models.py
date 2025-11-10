# backend/app/backtest/models.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class BacktestBar(BaseModel):
    """Barra de precio para backtesting"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }

class BacktestTrade(BaseModel):
    """Trade ejecutado en backtest"""
    id: str
    strategy_name: str
    symbol: str
    side: str  # BUY, SELL
    entry_timestamp: int
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float

    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def close_trade(
        self,
        exit_timestamp: int,
        exit_price: float,
        exit_reason: str,
        commission_rate: float = 0.0004
    ):
        """Cerrar trade y calcular PnL"""
        self.exit_timestamp = exit_timestamp
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calcular PnL
        if self.side == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.quantity

        # Restar comisiones (entry + exit)
        total_commission = (self.entry_price * self.quantity * commission_rate) + \
                          (exit_price * self.quantity * commission_rate)
        self.commission = total_commission
        self.pnl -= total_commission

        # Calcular PnL %
        invested = self.entry_price * self.quantity
        self.pnl_pct = (self.pnl / invested) * 100 if invested > 0 else 0

class BacktestResult(BaseModel):
    """Resultado completo de backtest"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float

    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Performance
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Trade metrics
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win: float
    largest_loss: float

    # Time metrics
    avg_trade_duration_minutes: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Trades detallados
    trades: List[BacktestTrade]