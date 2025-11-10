"""
Metrics calculation utilities - corrected version
Fixes:
- MaxDD calculated on equity curve (not % on notional)
- Stable Sharpe calculation with proper annualization
"""
import numpy as np
import pandas as pd


def max_drawdown(equity):
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity: pd.Series or np.array of equity values
        
    Returns:
        float: Maximum drawdown as negative decimal (e.g., -0.15 for 15% DD)
    """
    if isinstance(equity, list):
        equity = pd.Series(equity)
    
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()  # negative value


def sharpe_ratio(equity, periods_per_year=252, rf=0.0):
    """
    Calculate annualized Sharpe ratio from equity curve.
    
    Args:
        equity: pd.Series or np.array of equity values
        periods_per_year: int, number of periods per year (252 for daily, 252*390 for minute)
        rf: float, risk-free rate (default 0.0)
        
    Returns:
        float: Annualized Sharpe ratio
    """
    if isinstance(equity, list):
        equity = pd.Series(equity)
    
    returns = equity.pct_change().dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    sharpe = (mean_return - rf) / std_return * np.sqrt(periods_per_year)
    return sharpe


def calculate_metrics(trades_df, initial_equity=10000):
    """
    Calculate comprehensive trading metrics from trades dataframe.
    
    Args:
        trades_df: DataFrame with 'pnl' column
        initial_equity: float, starting equity
        
    Returns:
        dict: Metrics including PF, WinRate, Sharpe, MaxDD, Expectancy
    """
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0
        }
    
    # Build equity curve
    equity = [initial_equity]
    for pnl in trades_df['pnl']:
        equity.append(equity[-1] + pnl)
    equity = pd.Series(equity)
    
    # Win/Loss stats
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else (float('inf') if total_wins > 0 else 0.0)
    
    # Sharpe (assuming daily aggregation for stability)
    sharpe = sharpe_ratio(equity, periods_per_year=252)
    
    # MaxDD on equity curve
    max_dd = max_drawdown(equity)
    
    # Expectancy
    expectancy = trades_df['pnl'].mean() if len(trades_df) > 0 else 0.0
    
    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'expectancy': expectancy,
        'total_pnl': trades_df['pnl'].sum()
    }
