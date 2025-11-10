# backend/scripts/test_tp_sl.py

import asyncio
import sys
from pathlib import Path
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backtest.data_loader import BinanceHistoricalDataLoader
from app.backtest.backtest_engine import BacktestEngine
from app.backtest.performance_analytics import PerformanceAnalytics
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.momentum_scalping import MomentumScalpingStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.utils.logger import get_logger

logger = get_logger("test_tp_sl")

async def main():
    """Test TP/SL implementation"""
    
    print("=" * 80)
    print("TESTING TP/SL IMPLEMENTATION - Sprint V2.1.1")
    print("=" * 80)
    
    symbol = "BTCUSDT"
    
    # Cargar datos de 5 días REALES (1.2 GB collection)
    print("\n1️⃣ Loading 5-day REAL data with advanced features...")
    
    data_path = Path(f"data/advanced_features_5d/{symbol}_advanced_features.parquet")
    
    if not data_path.exists():
        print(f"   ❌ Data file not found: {data_path}")
        print("   Please ensure 5-day features are computed")
        return
    
    df = pd.read_parquet(data_path)
    print(f"   📊 Loaded {len(df):,} samples from {data_path.name}")
    print(f"   📁 Size: {data_path.stat().st_size / (1024**2):.1f} MB")
    
    # Mostrar columnas disponibles
    print(f"   📋 Features: {len(df.columns)} columns")
    print(f"      - DepthImb_L2L5, KyleLambda, TradeIntensity, VPIN, OFI, etc.")
    
    # Crear barras OHLCV sintéticas desde los snapshots
    # Agrupamos por minuto para simular barras 1m
    print("\n   🔄 Creating synthetic OHLCV bars from snapshots...")
    
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['minute'] = df['timestamp_dt'].dt.floor('1min')
    
    # Agrupar por minuto
    bars_df = df.groupby('minute').agg({
        'midprice': ['first', 'max', 'min', 'last'],  # OHLC
        'timestamp': 'first',
        'DepthImb_L2L5': 'mean',  # Promedio del feature principal
        'KyleLambda_1s': 'mean',
        'TradeIntensity_1s': 'mean',
        'RollImpact_1s': 'mean'
    }).reset_index()
    
    # Renombrar columnas
    bars_df.columns = ['_'.join(col).strip('_') for col in bars_df.columns.values]
    bars_df.rename(columns={
        'midprice_first': 'open',
        'midprice_max': 'high',
        'midprice_min': 'low',
        'midprice_last': 'close',
        'timestamp_first': 'timestamp'
    }, inplace=True)
    
    # Añadir volumen sintético (constante por ahora)
    bars_df['volume'] = 1.0
    
    # Asegurar que timestamp es int64 y está en milisegundos (no nanosegundos)
    # Los timestamps originales están en ms, así que solo convertimos a int
    bars_df['timestamp'] = bars_df['timestamp'].astype('int64') // 1_000_000  # ns -> ms
    
    # Convertir a lista de diccionarios
    bars = bars_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    
    print(f"   ✅ Created {len(bars):,} bars from {len(df):,} snapshots")
    print(f"   📅 Period: {bars_df['minute'].min()} to {bars_df['minute'].max()}")
    
    # Crear estrategia
    print("\n2️⃣ Creating Momentum strategy with TP/SL...")
    
    indicator_manager = IndicatorManager()
    indicator_manager.initialize_symbol(symbol)
    
    position_sizer = PositionSizer(account_balance=10000)
    signal_validator = SignalValidator(indicator_manager=indicator_manager)
    
    strategy = MomentumScalpingStrategy(
        symbol=symbol,
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator
    )
    
    # Backtest con TP/SL mejorado
    print("\n3️⃣ Running backtest with ATR-based TP/SL...")
    print("   Configuration:")
    print("   - Stop Loss: 1.5x ATR")
    print("   - Take Profit: 2.0x ATR")
    print("   - Commission: 0.04%")
    print("   - Slippage: 0.05%")
    
    engine = BacktestEngine(
        initial_balance=10000.0,
        commission_rate=0.0004,  # 0.04% Binance
        slippage_pct=0.05  # 0.05% slippage
    )
    
    # Configurar multipliers para TP/SL
    engine.atr_stop_multiplier = 1.5
    engine.tp_sl_ratio = 2.0
    
    result = await engine.run_backtest(
        strategy=strategy,
        bars=bars,
        symbol=symbol
    )
    
    # Análisis
    print("\n4️⃣ Analyzing results...")
    
    analytics = PerformanceAnalytics()
    analysis = analytics.analyze(result)
    
    # Imprimir reporte
    analytics.print_report(analysis)
    
    # Guardar
    filename = analytics.save_report(
        analysis,
        filename="reports/v2.1.1_tp_sl_test.json"
    )
    
    print(f"\n✅ Report saved: {filename}")
    
    # Métricas clave
    print("\n" + "=" * 80)
    print("KEY METRICS - Sprint V2.1.1")
    print("=" * 80)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Win Rate:         {result.win_rate:.2f}%")
    print(f"Profit Factor:    {result.profit_factor:.4f}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.4f}")
    print(f"Total P&L:        ${result.total_pnl:.2f}")
    print(f"Final Balance:    ${result.final_balance:.2f}")
    print(f"Expectancy:       ${analysis['expectancy']['expectancy']:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    
    # Verificar mejora vs baseline
    print("\n" + "=" * 80)
    print("📊 COMPARISON vs BASELINE")
    print("=" * 80)
    print("Baseline (PF from monetization 5d):  2.45")
    print(f"Current (TP/SL ATR-based):           {result.profit_factor:.4f}")
    
    if result.profit_factor >= 2.45:
        improvement = ((result.profit_factor - 2.45) / 2.45) * 100
        print(f"\n🎉 IMPROVEMENT: +{improvement:.2f}%")
        print("✅ TP/SL implementation is BETTER than baseline!")
    elif result.profit_factor >= 0.8:
        degradation = ((2.45 - result.profit_factor) / 2.45) * 100
        print(f"\n⚠️  DEGRADATION: -{degradation:.2f}%")
        print("🟡 TP/SL implementation is ACCEPTABLE (PF > 0.8)")
    else:
        print(f"\n❌ FAIL: PF {result.profit_factor:.4f} < 0.8 threshold")
        print("Need parameter adjustment or different approach")
    
    # Trade distribution
    if result.total_trades > 0:
        print("\n" + "=" * 80)
        print("TRADE DISTRIBUTION")
        print("=" * 80)
        print(f"Winners:          {result.winning_trades} ({result.win_rate:.1f}%)")
        print(f"Losers:           {result.losing_trades} ({100-result.win_rate:.1f}%)")
        print(f"Avg Win:          ${result.avg_win:.2f}")
        print(f"Avg Loss:         ${result.avg_loss:.2f}")
        print(f"Win/Loss Ratio:   {result.avg_win/result.avg_loss:.2f}" if result.avg_loss > 0 else "N/A")
        print(f"Largest Win:      ${result.largest_win:.2f}")
        print(f"Largest Loss:     ${result.largest_loss:.2f}")
        print(f"Avg Duration:     {result.avg_trade_duration_minutes:.1f} minutes")
    
    # Decision
    print("\n" + "=" * 80)
    print("🚨 SPRINT V2.1.1 VERDICT")
    print("=" * 80)
    
    if result.profit_factor >= 0.8 and result.total_trades >= 10:
        print("✅ PASS - TP/SL implementation working")
        print("📝 Next: Sprint V2.1.2 - Implement Trailing Stops")
    elif result.total_trades < 10:
        print("⚠️  INSUFFICIENT DATA - Need more trades")
        print("📝 Action: Increase signal frequency or extend backtest period")
    else:
        print("❌ FAIL - Parameters need adjustment")
        print("📝 Action: Try different SL/TP multipliers")
        print(f"   Suggested: SL=1.8x ATR, TP=2.5x ATR")

if __name__ == "__main__":
    asyncio.run(main())
