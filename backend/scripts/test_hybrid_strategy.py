# backend/scripts/test_hybrid_strategy.py

"""
Test Hybrid Momentum + Mean Reversion Strategy
Sprint V2.2.1

Objetivo: Validar estrategia híbrida con indicadores técnicos como señal primaria
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from app.backtest.backtest_engine import BacktestEngine
from app.core.strategies.hybrid_momentum_meanrev import HybridMomentumMeanRevStrategy
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores técnicos para la estrategia"""
    
    print("\n📊 Calculando indicadores técnicos...")
    
    # RSI (7)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['rsi_7'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (15, 2.0)
    df['bb_middle'] = df['close'].rolling(window=15).mean()
    bb_std = df['close'].rolling(window=15).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2.0)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2.0)
    
    # MACD (8, 17, 9)
    exp1 = df['close'].ewm(span=8, adjust=False).mean()
    exp2 = df['close'].ewm(span=17, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ADX (14)
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx_14'] = dx.rolling(window=14).mean()
    
    # Volume SMA (20)
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    
    print(f"✅ Indicadores calculados")
    print(f"   RSI range: {df['rsi_7'].min():.1f} - {df['rsi_7'].max():.1f}")
    print(f"   ADX range: {df['adx_14'].min():.1f} - {df['adx_14'].max():.1f}")
    
    return df

def resample_to_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Resamplear a barras de 5 minutos usando midprice como proxy"""
    
    print("\n📈 Resampling a 5-minutos...")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Usar midprice como proxy para OHLCV
    bars = df.resample('5min').agg({
        'midprice': ['first', 'max', 'min', 'last', 'count'],
        'DepthImb_L2L5': 'mean',
        'spread': 'mean',
    }).dropna()
    
    # Renombrar columnas
    bars.columns = ['open', 'high', 'low', 'close', 'volume', 'DepthImb_L2L5', 'spread']
    
    # Convertir spread a bps (ya está en valor absoluto)
    bars['Spread_bps'] = bars['spread'] / bars['close'] * 10000
    
    # Calcular volatilidad como rango high-low / close
    bars['VolatilityScore'] = (bars['high'] - bars['low']) / bars['close']
    
    bars.reset_index(inplace=True)
    
    print(f"✅ {len(bars)} barras de 5-minutos creadas")
    
    return bars

async def main():
    """Ejecutar backtest de estrategia híbrida"""
    
    print("=" * 80)
    print("SPRINT V2.3: 5-MINUTE TIMEFRAME - FUNDAMENTAL PIVOT")
    print("=" * 80)
    
    print("\n🔄 CAMBIO CRÍTICO: 1-min → 5-min")
    print("   🎯 Razón: Movimientos de precio en 1-min demasiado pequeños")
    print("   📊 TP 0.60% es alcanzable en 5-min (movimientos 0.25-0.60%)")
    print("   🚀 Menos ruido, señales más significativas")
    
    # Cargar datos
    data_file = Path("../data/advanced_features_5d/BTCUSDT_advanced_features.parquet")
    
    if not data_file.exists():
        print(f"❌ No se encontró archivo: {data_file}")
        return
    
    print(f"\n📂 Cargando {data_file.name}...")
    df = pd.read_parquet(data_file)
    print(f"✅ {len(df):,} registros cargados")
    print(f"   Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Resamplear a 1 minuto
    bars = resample_to_bars(df)
    
    # Calcular indicadores
    bars = calculate_indicators(bars)
    
    # Eliminar NaNs de los primeros valores
    bars = bars.dropna()
    print(f"\n✅ {len(bars):,} barras válidas para backtest")
    
    # Configurar estrategia
    strategy_config = {
        "rsi_period": 7,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bb_period": 15,
        "bb_std": 2.0,
        "bb_touch_threshold": 0.005,
        "macd_fast": 8,
        "macd_slow": 17,
        "macd_signal": 9,
        "volume_multiplier": 0,  # ❌ DESACTIVADO - orderbook data sin volumen real
        "adx_period": 14,
        "adx_min": 15,  # ✅ RELAJADO 20→15
        # OrderBook filters REMOVED
        "tp_pct": 0.60,  # ✅ AJUSTADO para 5-min (V2.3)
        "sl_pct": 0.25,  # ✅ AJUSTADO para 5-min (ratio 2.4:1)
        "trailing_activation_pct": 0.35,
        "trailing_distance_pct": 0.12,
        "use_time_limit": False,
        "risk_per_trade_pct": 0.8,
        "max_open_positions": 3,
    }
    
    print("\n🎯 Filosofía: Pure technical indicators en timeframe 5-min")
    print("📊 Cambios vs V2.2.2 (1-min):")
    print("   - ⏱️  Timeframe: 1-min → 5-min (CAMBIO FUNDAMENTAL)")
    print("   - 💰 TP/SL: 0.50%/0.20% → 0.60%/0.25%")
    print("   - ✅ Estrategia: RSI + BB + ADX (sin cambios)")
    print("   - 🎯 Objetivo: 50-120 trades con TP alcanzable")
    
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"   RSI(7): {strategy_config['rsi_oversold']} / {strategy_config['rsi_overbought']}")
    print(f"   BB(15): {strategy_config['bb_period']} períodos, {strategy_config['bb_std']} std")
    print(f"   MACD: {strategy_config['macd_fast']}/{strategy_config['macd_slow']}/{strategy_config['macd_signal']}")
    print(f"   Volume: ❌ DESACTIVADO (orderbook data)")
    print(f"   ADX: >{strategy_config['adx_min']}")
    print(f"   OrderBook: ❌ ELIMINADO")
    print(f"   TP/SL: {strategy_config['tp_pct']}% / {strategy_config['sl_pct']}% (ratio 2.4:1 - 5-min optimized)")
    print(f"   Time Limit: {'Sí' if strategy_config['use_time_limit'] else '❌ DESACTIVADO'}")
    print(f"   Max Posiciones: {strategy_config['max_open_positions']}")
    
    # Configurar backtest
    backtest_config = {
        "initial_balance": 10000,
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "slippage": 0.0001
    }
    
    # Crear componentes necesarios
    print("\n⚙️  Inicializando componentes...")
    indicator_manager = IndicatorManager()
    position_sizer = PositionSizer(account_balance=backtest_config['initial_balance'])
    signal_validator = SignalValidator(indicator_manager=indicator_manager)
    
    # Crear estrategia
    strategy = HybridMomentumMeanRevStrategy(
        symbol="BTCUSDT",
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator,
        config=strategy_config
    )
    
    print(f"✅ Estrategia: {strategy.name}")
    print(f"✅ Components: IndicatorManager, PositionSizer, SignalValidator")
    
    # Crear motor de backtest
    engine = BacktestEngine(
        initial_balance=backtest_config['initial_balance'],
        commission_rate=backtest_config['maker_fee'],
        slippage_pct=backtest_config['slippage'] * 100,  # Convertir a porcentaje
        max_open_positions=strategy_config['max_open_positions']
    )
    
    print("\n" + "=" * 80)
    print("EJECUTANDO BACKTEST...")
    print("=" * 80)
    
    # Convertir bars a formato esperado (lista de dicts con timestamp en ms)
    bars_list = []
    for _, row in bars.iterrows():
        bar_dict = {
            'timestamp': int(row['timestamp'].timestamp() * 1000),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'DepthImb_L2L5': row['DepthImb_L2L5'],
            'Spread_bps': row['Spread_bps'],
            'VolatilityScore': row['VolatilityScore'],
            'rsi_7': row['rsi_7'],
            'bb_upper': row['bb_upper'],
            'bb_middle': row['bb_middle'],
            'bb_lower': row['bb_lower'],
            'macd': row['macd'],
            'macd_signal': row['macd_signal'],
            'macd_histogram': row['macd_histogram'],
            'adx_14': row['adx_14'],
            'volume_sma_20': row['volume_sma_20'],
        }
        bars_list.append(bar_dict)
    
    # Ejecutar backtest (es async)
    result = await engine.run_backtest(
        strategy=strategy,
        bars=bars_list,
        symbol="BTCUSDT"
    )
    
    # Calcular métricas adicionales
    roi = (result.final_balance - result.initial_balance) / result.initial_balance * 100
    expectancy = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
    
    # Contar exit reasons
    exit_reasons = {}
    for trade in result.trades:
        reason = trade.exit_reason or "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS V2.3")
    print("=" * 80)
    
    print(f"\n💰 BALANCE:")
    print(f"   Inicial:  ${result.initial_balance:,.2f}")
    print(f"   Final:    ${result.final_balance:,.2f}")
    print(f"   P&L:      ${result.total_pnl:+,.2f}")
    print(f"   ROI:      {roi:+.2f}%")
    
    print(f"\n📈 TRADES:")
    print(f"   Total:        {result.total_trades}")
    print(f"   Ganadores:    {result.winning_trades} ({result.win_rate:.2f}%)")
    print(f"   Perdedores:   {result.losing_trades}")
    print(f"   Expectancy:   ${expectancy:+.2f}")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    
    print(f"\n📊 POR EXIT REASON:")
    for reason, count in exit_reasons.items():
        pct = count / result.total_trades * 100 if result.total_trades > 0 else 0
        print(f"   {reason:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Diagnostics de estrategia
    diagnostics = strategy.get_diagnostics()
    
    print(f"\n🔍 DIAGNÓSTICO DE FILTROS:")
    print(f"   Barras procesadas:     {diagnostics['bars_processed']:,}")
    print(f"\n   Señales potenciales:")
    print(f"      RSI:   {diagnostics['potential_signals']['rsi']:,}")
    print(f"      BB:    {diagnostics['potential_signals']['bb']:,}")
    print(f"      MACD:  {diagnostics['potential_signals']['macd']:,}")
    print(f"      Total: {diagnostics['potential_signals']['total']:,}")
    print(f"\n   Bloqueadas por:")
    print(f"      Volume:    {diagnostics['filtered_by']['volume']:,}")
    print(f"      ADX:       {diagnostics['filtered_by']['adx']:,}")
    print(f"      OrderBook: {diagnostics['filtered_by']['orderbook']:,}")
    print(f"      Total:     {diagnostics['filtered_by']['total']:,}")
    print(f"\n   Señales generadas:  {diagnostics['signals_generated']}")
    print(f"   Tasa de conversión: {diagnostics['conversion_rate']:.2f}%")
    
    # Comparación con V2.2.2
    print(f"\n📊 COMPARACIÓN: V2.2.2 (1-min) vs V2.3 (5-min)")
    print("=" * 80)
    print(f"{'Métrica':<20} {'V2.2.2 (1-min)':<25} {'V2.3 (5-min)':<25} {'Cambio':<15}")
    print("-" * 80)
    
    v222_metrics = {
        'Trades': 12,
        'Win Rate': 50.00,
        'Expectancy': -17.48,
        'Profit Factor': 0.72,
        'ROI': -2.10,
        'Final Balance': 9790.18
    }
    
    v223_metrics = {
        'Trades': result.total_trades,
        'Win Rate': result.win_rate,
        'Expectancy': expectancy,
        'Profit Factor': result.profit_factor,
        'ROI': roi,
        'Final Balance': result.final_balance
    }
    
    for metric in ['Trades', 'Win Rate', 'Expectancy', 'Profit Factor', 'ROI']:
        v222 = v222_metrics[metric]
        v223 = v223_metrics[metric]
        change = v223 - v222
        change_pct = (change / v222 * 100) if v222 != 0 else 0
        
        print(f"{metric:<20} {v222:>24.2f}  {v223:>24.2f}  {change:+14.2f} ({change_pct:+.1f}%)")
    
    # Evaluación de criterios de éxito
    print(f"\n✅ CRITERIOS DE ÉXITO (Sprint V2.3):")
    print("=" * 80)
    
    criteria = {
        "Trades >= 50": result.total_trades >= 50,
        "Win Rate >= 45%": result.win_rate >= 45,
        "Expectancy > $0.40": expectancy > 0.40,
        "Profit Factor >= 1.2": result.profit_factor >= 1.2,
        "ROI > 3%": roi > 3
    }
    
    met = sum(criteria.values())
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}  {criterion}")
    
    print(f"\n📊 RESULTADO: {met}/5 criterios cumplidos")
    
    if met >= 4:
        decision = "✅ EXITOSO - Proceder a V2.3.1 (Optimización de parámetros)"
    elif met == 3:
        decision = "⚙️  PROMETEDOR - Probar 15-min timeframe o ajustes menores"
    elif met == 2:
        decision = "🔧 MARGINAL - Probar 15-min timeframe como V2.3.1"
    else:
        decision = "❌ FALLIDO - Pivotar a V2.4 (estrategia completamente diferente)"
    
    print(f"\n🎯 DECISIÓN: {decision}")
    
    # Guardar resultados
    report_file = Path("reports/v2.3_5min_timeframe.json")
    report_file.parent.mkdir(exist_ok=True)
    
    report = {
        "sprint": "V2.3",
        "strategy": "5-Minute Timeframe Pivot",
        "timeframe": "5min",
        "timestamp": datetime.now().isoformat(),
        "config": strategy_config,
        "backtest_config": backtest_config,
        "results": result.model_dump(),  # Pydantic v2
        "expectancy": expectancy,
        "roi": roi,
        "exit_reasons": exit_reasons,
        "diagnostics": diagnostics,
        "criteria_met": met,
        "decision": decision,
        "comparison": {
            "v2.2.2": v222_metrics,
            "v2.3": v223_metrics
        }
    }
    
    # Convertir datetime objects a strings
    def convert_datetimes(obj):
        if isinstance(obj, dict):
            return {k: convert_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetimes(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    report = convert_datetimes(report)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Reporte guardado: {report_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
