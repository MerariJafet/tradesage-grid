# backend/scripts/test_orderbook_strategy.py

import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Imports del sistema
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backtest.backtest_engine import BacktestEngine
from app.backtest.performance_analytics import PerformanceAnalytics
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.orderbook_imbalance import OrderBookImbalanceStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator

async def main():
    """
    Test de Order Book Imbalance Strategy con datos reales de 5 días (3.4M snapshots)
    """
    
    print("""
====================================================================================
                   SPRINT V2.1.3 - MULTI-POSITION ENGINE TEST
====================================================================================
""")
    
    symbol = "BTCUSDT"
    
    # ================================================
    # PASO 1: CARGAR DATOS CON ORDER BOOK FEATURES
    # ================================================
    
    print("\n📊 PASO 1: Cargando datos de 5 días con features de order book...")
    
    data_path = Path('data/advanced_features_5d/BTCUSDT_advanced_features.parquet')
    
    if not data_path.exists():
        print(f"❌ ERROR: Archivo no encontrado: {data_path}")
        print("   Verifica que el path sea correcto.")
        return
    
    df = pd.read_parquet(data_path)
    
    print(f"   ✅ Cargados {len(df):,} snapshots")
    print(f"   📁 Tamaño: {data_path.stat().st_size / (1024**2):.1f} MB")
    print(f"   📅 Período: {pd.to_datetime(df['timestamp'].min(), unit='ms')} a {pd.to_datetime(df['timestamp'].max(), unit='ms')}")
    
    # Verificar features críticos
    required_features = ['DepthImb_L2L5', 'spread', 'TradeIntensity_1s', 'KyleLambda_1s', 'midprice']
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        print(f"❌ ERROR: Features faltantes: {missing}")
        print(f"   Disponibles: {list(df.columns)}")
        return
    
    print(f"   ✅ Todos los features requeridos presentes")
    print(f"   📋 Features: DepthImb_L2L5, spread, TradeIntensity_1s, KyleLambda_1s")
    
    # ================================================
    # PASO 2: RESAMPLE A 1 MINUTO
    # ================================================
    
    print("\n⏱️  PASO 2: Resampling a barras de 1 minuto...")
    
    # Convertir timestamp
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['minute'] = df['timestamp_dt'].dt.floor('1min')
    
    # Agrupar por minuto agregando features
    bars_df = df.groupby('minute').agg({
        'midprice': ['first', 'max', 'min', 'last'],  # OHLC
        'timestamp': 'first',
        'DepthImb_L2L5': 'mean',       # Promedio del imbalance
        'spread': 'mean',              # Promedio del spread
        'TradeIntensity_1s': 'mean',   # Promedio trade intensity
        'KyleLambda_1s': 'mean',       # Promedio kyle lambda
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
    
    # Añadir volumen sintético
    bars_df['volume'] = 1.0
    
    # Asegurar que timestamp es int64 en milisegundos
    bars_df['timestamp'] = bars_df['timestamp'].astype('int64')
    
    # Limpiar NaN
    bars_df = bars_df.dropna()
    
    print(f"   ✅ Creadas {len(bars_df):,} barras de 1 minuto desde {len(df):,} snapshots")
    print(f"   📊 Ratio: {len(df) / len(bars_df):.0f} snapshots por barra")
    
    # ================================================
    # PASO 3: CONVERTIR A FORMATO DE BACKTEST
    # ================================================
    
    print("\n🔄 PASO 3: Preparando datos para backtest...")
    
    # Convertir a lista de diccionarios
    bars = []
    for _, row in bars_df.iterrows():
        # Convertir timestamp a milisegundos (int)
        ts_ms = int(row['timestamp']) // 1_000_000  # nanosegundos -> milisegundos
        
        bar_dict = {
            'timestamp': ts_ms,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'DepthImb_L2L5': float(row['DepthImb_L2L5_mean']),
            'spread': float(row['spread_mean']),
            'TradeIntensity_1s': float(row['TradeIntensity_1s_mean']),
            'KyleLambda_1s': float(row['KyleLambda_1s_mean']),
        }
        bars.append(bar_dict)
    
    print(f"   ✅ {len(bars):,} barras listas para backtest")
    
    # Mostrar estadísticas de DepthImb_L2L5
    imbalance_values = [b['DepthImb_L2L5'] for b in bars]
    print(f"\n   📊 Estadísticas DepthImb_L2L5:")
    print(f"      Min: {min(imbalance_values):.4f}")
    print(f"      Max: {max(imbalance_values):.4f}")
    print(f"      Mean: {np.mean(imbalance_values):.4f}")
    print(f"      Std: {np.std(imbalance_values):.4f}")
    print(f"      > +0.15 (old threshold): {sum(1 for v in imbalance_values if v > 0.15)} bars")
    print(f"      < -0.15 (old threshold): {sum(1 for v in imbalance_values if v < -0.15)} bars")
    print(f"      > +0.10 (new threshold): {sum(1 for v in imbalance_values if v > 0.10)} bars")
    print(f"      < -0.10 (new threshold): {sum(1 for v in imbalance_values if v < -0.10)} bars")
    
    # ================================================
    # PASO 4: CREAR ESTRATEGIA
    # ================================================
    
    print("\n🎯 PASO 4: Inicializando estrategia OrderBook Imbalance...")
    
    indicator_manager = IndicatorManager()
    position_sizer = PositionSizer(account_balance=10000)
    signal_validator = SignalValidator(indicator_manager=indicator_manager)
    
    # Configuración de la estrategia - V2.1.2.2 KYLE_LAMBDA REMOVIDO
    strategy_config = {
        "imbalance_long_threshold": 0.10,     # 0.15 → 0.10 (más señales)
        "imbalance_short_threshold": -0.10,   # -0.15 → -0.10 (más señales)
        "min_trade_intensity": 0.3,           # 0.5 → 0.3 (menos restrictivo)
        "max_spread": 3.0,                    # 1.0 → 3.0 (menos restrictivo para BTC)
        "max_kyle_lambda": 999.0,             # REMOVIDO: era el filtro más restrictivo (93.1%)
        "tp_pct": 0.30,
        "sl_pct": 0.15,
        "trailing_activation_pct": 0.20,
        "trailing_distance_pct": 0.08,
    }
    
    strategy = OrderBookImbalanceStrategy(
        symbol=symbol,
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator,
        config=strategy_config
    )
    
    config_summary = strategy.get_config_summary()
    print(f"   ✅ Estrategia: {config_summary['strategy']} - V2.1.2.1 (FILTROS RELAJADOS)")
    print(f"   📋 Thresholds: Long={config_summary['thresholds']['long']}, Short={config_summary['thresholds']['short']} (0.15 → 0.10)")
    print(f"   💰 TP/SL: {config_summary['exits']['tp']}/{config_summary['exits']['sl']}")
    print(f"   🔧 Filters: Spread<{config_summary['filters']['max_spread']} (1.0→3.0), TI>{config_summary['filters']['min_trade_intensity']} (0.5→0.3)")
    print(f"   🎯 Target: 50-150 trades for statistical validation")
    
    # ================================================
    # PASO 5: EJECUTAR BACKTEST
    # ================================================
    
    print("\n🚀 PASO 5: Ejecutando backtest con datos de 5 días...")
    print(f"   Balance inicial: $10,000")
    print(f"   Commission rate: 0.02% (MAKER)")
    print(f"   Slippage: 0.05%")
    print(f"   Procesando {len(bars):,} barras...\n")
    
    engine = BacktestEngine(
        initial_balance=10000.0,
        commission_rate=0.0002,  # MAKER fees 0.02%
        slippage_pct=0.05,       # Slippage realista
        max_open_positions=3     # 🚀 NUEVO: Permitir 3 posiciones simultáneas
    )
    
    # Ejecutar backtest
    result = await engine.run_backtest(
        strategy=strategy,
        bars=bars,
        symbol=symbol
    )
    
    # ================================================
    # PASO 6: ANÁLISIS DE RESULTADOS
    # ================================================
    
    print("\n📈 PASO 6: Analizando performance...")
    
    analytics = PerformanceAnalytics()
    analysis = analytics.analyze(result)
    
    # Imprimir reporte completo
    analytics.print_report(analysis)
    
    # ================================================
    # PASO 6.5: DIAGNÓSTICO DE FILTRADO (NUEVO)
    # ================================================
    
    print("\n" + "="*84)
    print("🔍 FILTER DIAGNOSTICS - SIGNAL FUNNEL ANALYSIS")
    print("="*84)
    
    filter_stats = strategy.get_filter_stats()
    
    print(f"\n📊 SIGNAL FUNNEL:")
    print(f"   Total bars processed: {filter_stats['total_bars']:,}")
    print(f"\n   🎯 POTENTIAL SIGNALS (before filters):")
    print(f"      LONG signals (DepthImb > {strategy.config['imbalance_long_threshold']}):  {filter_stats['potential_long_signals']:,}")
    print(f"      SHORT signals (DepthImb < {strategy.config['imbalance_short_threshold']}): {filter_stats['potential_short_signals']:,}")
    print(f"      TOTAL potential: {filter_stats['potential_long_signals'] + filter_stats['potential_short_signals']:,}")
    
    total_potential = filter_stats['potential_long_signals'] + filter_stats['potential_short_signals']
    
    if total_potential > 0:
        print(f"\n   🚫 FILTERED OUT:")
        print(f"      By SPREAD filter (>{strategy.config['max_spread']}): {filter_stats['filtered_by_spread']:,} ({filter_stats['filtered_by_spread_pct']:.1f}%)")
        print(f"      By TRADE_INTENSITY filter (<{strategy.config['min_trade_intensity']}): {filter_stats['filtered_by_trade_intensity']:,} ({filter_stats['filtered_by_trade_intensity_pct']:.1f}%)")
        print(f"      By KYLE_LAMBDA filter (>{strategy.config['max_kyle_lambda']}): {filter_stats['filtered_by_kyle_lambda']:,} ({filter_stats['filtered_by_kyle_lambda_pct']:.1f}%)")
        
        print(f"\n   ✅ PASSED ALL FILTERS: {filter_stats['passed_all_filters']:,} ({filter_stats['passed_all_filters_pct']:.1f}%)")
        print(f"   📈 SIGNALS GENERATED: {filter_stats['actual_signals_generated']:,} ({filter_stats['conversion_rate']:.2f}%)")
        
        # Identificar el filtro más restrictivo
        filter_impacts = [
            ("SPREAD", filter_stats['filtered_by_spread'], filter_stats['filtered_by_spread_pct']),
            ("TRADE_INTENSITY", filter_stats['filtered_by_trade_intensity'], filter_stats['filtered_by_trade_intensity_pct']),
            ("KYLE_LAMBDA", filter_stats['filtered_by_kyle_lambda'], filter_stats['filtered_by_kyle_lambda_pct']),
        ]
        filter_impacts.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n   ⚠️ MOST RESTRICTIVE FILTER: {filter_impacts[0][0]} (eliminates {filter_impacts[0][2]:.1f}% of potential signals)")
        
        if filter_impacts[0][2] > 70:
            print(f"      🔧 RECOMMENDATION: Consider relaxing or removing {filter_impacts[0][0]} filter")
    
    print("="*84)
    
    # ================================================
    # PASO 6.6: SIGNAL BLOCKING ANALYSIS (V2.1.3)
    # ================================================
    
    print("\n" + "="*90)
    print(" " * 30 + "🚀 SIGNAL BLOCKING ANALYSIS")
    print("="*90)
    
    if hasattr(engine, 'total_signals_generated'):
        print(f"\n📊 SIGNAL STATISTICS:")
        print(f"   Total signals generated: {engine.total_signals_generated}")
        print(f"   Total signals blocked: {engine.total_signals_blocked}")
        
        if engine.total_signals_generated > 0:
            block_rate = (engine.total_signals_blocked / engine.total_signals_generated) * 100
            print(f"   Block rate: {block_rate:.1f}%")
            
            if block_rate < 20:
                print("\n   ✅ LOW BLOCKING RATE - Engine is efficient")
                print(f"      Max {engine.max_open_positions} positions is sufficient")
            elif block_rate < 50:
                print("\n   ⚠️  MODERATE BLOCKING - Some opportunities missed")
                print(f"      Consider increasing max_positions from {engine.max_open_positions} to {engine.max_open_positions + 2}")
            else:
                print("\n   ❌ HIGH BLOCKING RATE - Many opportunities missed!")
                print(f"      RECOMMENDATION: Increase max_positions from {engine.max_open_positions} to {engine.max_open_positions * 2}")
        
        # Comparación con V2.1.2.2
        print(f"\n📈 IMPROVEMENT VS V2.1.2.2 (Single Position):")
        print(f"   V2.1.2.2 Bars processed: 2 (0.03% of dataset)")
        print(f"   V2.1.3 Signals generated: {engine.total_signals_generated}")
        if engine.total_signals_generated > 2:
            improvement = engine.total_signals_generated / 2
            print(f"   Improvement: {improvement:.1f}x MORE opportunities processed ✅")
    
    print("="*90)
    
    # ================================================
    # PASO 7: GUARDAR RESULTADOS
    # ================================================
    
    print("\n💾 PASO 7: Guardando resultados...")
    
    filename = analytics.save_report(
        analysis,
        filename="reports/v2.1.3_multi_position.json"
    )
    
    print(f"   ✅ Reporte guardado: {filename}")
    
    # ================================================
    # PASO 8: COMPARACIÓN VS V2.1.1
    # ================================================
    
    print("\n" + "=" * 90)
    print(" " * 30 + "COMPARATIVE ANALYSIS")
    print("=" * 90)
    
    print("\n📊 V2.1.1 (Momentum Strategy - BASELINE):")
    print("   Total Trades: 2")
    print("   Win Rate: 50.0%")
    print("   Expectancy: -$40.17")
    print("   Profit Factor: 0.23")
    print("   ROI: -0.80%")
    
    print("\n📊 V2.1.2 (OrderBook - Restrictive Filters):")
    print("   Total Trades: 5")
    print("   Win Rate: 40.0%")
    print("   Expectancy: -$6.37")
    print("   Profit Factor: 0.87")
    print("   ROI: -0.32%")
    print("   Issues: Only 5 trades, filters too restrictive (99.87% filtered)")
    print("   Data: Same 5-day dataset (3.4M snapshots)")
    
    expectancy = analysis['expectancy']['expectancy']
    roi = ((result.final_balance - result.initial_balance) / result.initial_balance * 100)
    
    print(f"\n🚀 CURRENT (V2.1.2.1 - OrderBook Relaxed Filters):")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Win Rate: {result.win_rate:.2f}%")
    print(f"   Expectancy: ${expectancy:.2f}")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    print(f"   ROI: {roi:.2f}%")
    print(f"   Config: Imbalance ±0.10 (vs ±0.15), Spread<3.0 (vs <1.0), TI>0.3 (vs >0.5)")
    print(f"   Data: Same 5-day dataset (3.4M snapshots)")
    
    # Evaluación de mejoras
    print("\n" + "=" * 90)
    print(" " * 35 + "EVALUATION")
    print("=" * 90)
    
    # 1. Volumen de trades
    trade_improvement_vs_v211 = (result.total_trades / 2) if result.total_trades > 0 else 0
    trade_improvement_vs_v212 = (result.total_trades / 5) if result.total_trades > 0 else 0
    if result.total_trades >= 50:
        print(f"\n✅ TRADE VOLUME: Sufficient ({result.total_trades} trades) for statistical validity")
        print(f"   Improvement: {trade_improvement_vs_v211:.1f}x vs V2.1.1, {trade_improvement_vs_v212:.1f}x vs V2.1.2")
    elif result.total_trades >= 20:
        print(f"\n⚠️  TRADE VOLUME: Acceptable ({result.total_trades} trades) but could be higher")
        print(f"   Improvement: {trade_improvement_vs_v211:.1f}x vs V2.1.1, {trade_improvement_vs_v212:.1f}x vs V2.1.2")
        print("   Recommendation: Consider further relaxing thresholds or removing more filters")
    else:
        print(f"\n❌ TRADE VOLUME: Too low ({result.total_trades} trades)")
        print(f"   Improvement: {trade_improvement_vs_v211:.1f}x vs V2.1.1, {trade_improvement_vs_v212:.1f}x vs V2.1.2")
        print("   Action required: Further relax thresholds or remove filters entirely")
    
    # 2. Expectancy
    expectancy_improvement = expectancy - (-40.17)
    if expectancy > 0.40:
        print(f"\n✅ EXPECTANCY: EXCELLENT (${expectancy:.2f} per trade)")
        print(f"   Improvement: +${expectancy_improvement:.2f} vs baseline")
        print("   Strategy is profitable and ready for optimization")
    elif expectancy > 0:
        print(f"\n✅ EXPECTANCY: POSITIVE (${expectancy:.2f} per trade)")
        print(f"   Improvement: +${expectancy_improvement:.2f} vs baseline")
        print("   Strategy shows promise, optimization can improve")
    else:
        print(f"\n⚠️  EXPECTANCY: NEGATIVE (${expectancy:.2f} per trade)")
        print(f"   Change: ${expectancy_improvement:.2f} vs baseline")
        print("   Strategy needs parameter adjustment")
    
    # 3. Win Rate
    win_rate_improvement = result.win_rate - 50.0
    if result.win_rate >= 55:
        print(f"\n✅ WIN RATE: Strong ({result.win_rate:.1f}%)")
        print(f"   Improvement: +{win_rate_improvement:.1f}pp vs baseline")
    elif result.win_rate >= 50:
        print(f"\n✅ WIN RATE: Acceptable ({result.win_rate:.1f}%)")
        print(f"   Change: {win_rate_improvement:+.1f}pp vs baseline")
    else:
        print(f"\n⚠️  WIN RATE: Below target ({result.win_rate:.1f}%)")
        print(f"   Change: {win_rate_improvement:+.1f}pp vs baseline")
        print("   Need to improve TP/SL ratio or entry quality")
    
    # 4. Profit Factor
    pf_improvement = result.profit_factor - 0.23
    if result.profit_factor >= 1.5:
        print(f"\n✅ PROFIT FACTOR: Excellent ({result.profit_factor:.2f})")
        print(f"   Improvement: +{pf_improvement:.2f} vs baseline")
    elif result.profit_factor >= 1.2:
        print(f"\n✅ PROFIT FACTOR: Good ({result.profit_factor:.2f})")
        print(f"   Improvement: +{pf_improvement:.2f} vs baseline")
    elif result.profit_factor >= 1.0:
        print(f"\n⚠️  PROFIT FACTOR: Marginal ({result.profit_factor:.2f})")
        print(f"   Improvement: +{pf_improvement:.2f} vs baseline")
    else:
        print(f"\n⚠️  PROFIT FACTOR: Below 1.0 ({result.profit_factor:.2f})")
        print(f"   Improvement: +{pf_improvement:.2f} vs baseline")
    
    # Recomendaciones finales
    print("\n" + "=" * 90)
    print(" " * 30 + "NEXT STEPS RECOMMENDATION")
    print("=" * 90)
    
    success_score = 0
    if result.total_trades >= 50: success_score += 1
    if expectancy > 0: success_score += 1
    if result.win_rate >= 50: success_score += 1
    if result.profit_factor >= 1.0: success_score += 1
    
    if success_score >= 3:
        print("\n🎉 SUCCESS! Strategy is ready for:")
        print("   1. ✅ Parameter optimization (Sprint V2.2)")
        print("   2. ✅ Session filtering (London/NY)")
        print("   3. ✅ Maker-only orders implementation")
        print(f"\n   Score: {success_score}/4 criteria met")
    elif success_score >= 2:
        print("\n⚙️  OPTIMIZATION NEEDED:")
        print("   1. Relax imbalance thresholds (0.15 → 0.12)")
        print("   2. Adjust TP/SL ratio (current 2:1)")
        print("   3. Re-test with new parameters")
        print(f"\n   Score: {success_score}/4 criteria met")
    else:
        print("\n🔧 MAJOR ADJUSTMENT REQUIRED:")
        print("   1. Significantly relax thresholds (0.15 → 0.10)")
        print("   2. Remove or relax volume/spread filters")
        print("   3. Consider different imbalance features")
        print(f"\n   Score: {success_score}/4 criteria met")
    
    print("\n" + "=" * 90)
    print(" " * 35 + "SUMMARY TABLE")
    print("=" * 90)
    
    print(f"\n| Metric          | V2.1.1 | V2.1.2 | V2.1.2.1 | vs V2.1.1 | vs V2.1.2 | Status |")
    print(f"|-----------------|--------|--------|----------|-----------|-----------|--------|")
    print(f"| Trades          | 2      | 5      | {result.total_trades:<8} | {trade_improvement_vs_v211:>6.1f}x    | {trade_improvement_vs_v212:>6.1f}x    | {'✅' if result.total_trades >= 50 else '⚠️'} |")
    print(f"| Win Rate        | 50.0%  | 40.0%  | {result.win_rate:<7.1f}% | {result.win_rate - 50:>+6.1f}pp  | {result.win_rate - 40:>+6.1f}pp  | {'✅' if result.win_rate >= 50 else '⚠️'} |")
    print(f"| Expectancy      | -$40.17| -$6.37 | ${expectancy:<8.2f} | ${expectancy - (-40.17):>+8.2f} | ${expectancy - (-6.37):>+8.2f} | {'✅' if expectancy > 0 else '⚠️'} |")
    print(f"| Profit Factor   | 0.23   | 0.87   | {result.profit_factor:<8.2f} | {result.profit_factor - 0.23:>+8.2f} | {result.profit_factor - 0.87:>+8.2f} | {'✅' if result.profit_factor >= 1.0 else '⚠️'} |")
    print(f"| ROI             | -0.80% | -0.32% | {roi:<7.2f}% | {roi - (-0.80):>+6.2f}pp  | {roi - (-0.32):>+6.2f}pp  | {'✅' if roi > 0 else '⚠️'} |")
    
    print("\n" + "=" * 90)
    print("✅ TEST COMPLETED - Sprint V2.1.2.1 (FILTROS RELAJADOS)")
    print("=" * 90)
    print("\n🎯 Changes from V2.1.2:")
    print("   - Imbalance threshold: ±0.15 → ±0.10")
    print("   - Max spread: 1.0 → 3.0")
    print("   - Min trade intensity: 0.5 → 0.3")
    print("   - Max kyle lambda: 0.5 → 1.0")

if __name__ == "__main__":
    asyncio.run(main())
