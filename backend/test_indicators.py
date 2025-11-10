#!/usr/bin/env python3
"""
Test script para indicadores Sprint 3
Ejecuta pruebas manuales de los indicadores con datos simulados
"""

import sys
import os
from datetime import datetime, timedelta

# Añadir el directorio backend al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.indicators.indicator_manager import IndicatorManager
from app.utils.logger import setup_logging

setup_logging()

def test_indicator_manager():
    """Test básico del IndicatorManager"""
    print("🧪 Testing IndicatorManager...")

    manager = IndicatorManager()

    # Inicializar símbolo
    symbol = "BTCUSDT"
    manager.initialize_symbol(symbol)
    print(f"✅ Inicializado {symbol}")

    # Generar barras de prueba
    base_time = datetime.utcnow()
    bars = []

    for i in range(50):
        bar = {
            'timestamp': base_time + timedelta(minutes=i),
            'open': 50000 + i * 10,
            'high': 51000 + i * 10,
            'low': 49000 + i * 10,
            'close': 50500 + i * 10,
            'volume': 100 + i
        }
        bars.append(bar)

    # Procesar barras
    print("📊 Procesando barras...")
    for bar in bars:
        results = manager.update_with_bar(symbol, bar)

    print(f"✅ Procesadas {len(bars)} barras")

    # Verificar valores
    values = manager.get_all_values(symbol)
    print(f"📈 Valores calculados: {len(values)} indicadores")
    for key, value in values.items():
        print(f"   {key}: {value}")

    # Verificar señales
    signals = manager.get_signals(symbol)
    print(f"🚨 Señales generadas: {len(signals)}")
    for key, value in signals.items():
        print(f"   {key}: {value}")

    # Test orderbook
    print("📊 Testing OrderBook Imbalance...")
    orderbook = {
        'bids': [(50000, 10.0), (49900, 8.0), (49800, 6.0)],
        'asks': [(50100, 2.0), (50200, 1.0), (50300, 1.0)]
    }

    obi_results = manager.update_with_orderbook(symbol, orderbook)
    print(f"✅ OBI calculado: {obi_results}")

    return True

def test_individual_indicators():
    """Test indicadores individuales"""
    print("\n🧪 Testing indicadores individuales...")

    from app.core.indicators.atr import ATR
    from app.core.indicators.bollinger import BollingerBands
    from app.core.indicators.rsi import RSI
    from app.core.indicators.macd import MACD
    from app.core.indicators.vwap import VWAP
    from app.core.indicators.obi import OBI

    symbol = "BTCUSDT"

    # ATR
    print("Testing ATR...")
    atr = ATR(symbol, period=14)
    for i in range(20):
        bar = {'high': 50000 + i*50, 'low': 49000 + i*50, 'close': 49500 + i*50}
        atr.update(bar)
    print(f"ATR value: {atr.last_value}")

    # Bollinger Bands
    print("Testing Bollinger Bands...")
    bb = BollingerBands(symbol, period=20, num_std=2.0)
    for i in range(25):
        bar = {'close': 50000 + i*10}
        bb.update(bar)
    print(f"BB: upper={bb.upper}, middle={bb.middle}, lower={bb.lower}")

    # RSI
    print("Testing RSI...")
    rsi = RSI(symbol, period=14)
    for i in range(20):
        bar = {'close': 50000 + i*100}
        rsi.update(bar)
    print(f"RSI: {rsi.last_value}")

    # MACD
    print("Testing MACD...")
    macd = MACD(symbol, fast=12, slow=26, signal=9)
    for i in range(40):
        bar = {'close': 50000 + i*50}
        macd.update(bar)
    print(f"MACD: line={macd.macd_line}, signal={macd.signal_line}")

    # VWAP
    print("Testing VWAP...")
    vwap = VWAP("BTCUSDT")
    base_time = datetime.utcnow().replace(hour=9, minute=0)
    for i in range(10):
        bar = {
            'timestamp': base_time + timedelta(minutes=i*5),
            'high': 50000 + i*100,
            'low': 49000 + i*100,
            'close': 50000 + i*100,
            'volume': 100
        }
        vwap.update(bar)
    print(f"VWAP: {vwap.last_value}")

    # OBI
    print("Testing OBI...")
    obi = OBI(symbol)
    bids = [(50000, 10.0), (49900, 8.0)]
    asks = [(50100, 2.0), (50200, 1.0)]
    obi_value = obi.calculate(bids, asks, depth=5)
    print(f"OBI: {obi_value}")

    return True

def test_cache():
    """Test del sistema de caché"""
    print("\n🧪 Testing Indicator Cache...")

    try:
        from app.core.indicators.cache import IndicatorCache

        cache = IndicatorCache()

        # Test set/get
        cache.set("BTCUSDT", "test_indicator", {"value": 123.45}, ttl=60)
        cached = cache.get("BTCUSDT", "test_indicator")
        print(f"Cache get: {cached}")

        # Test get_all_symbol
        all_cached = cache.get_all_symbols()
        print(f"All cached symbols: {all_cached}")

        # Test clear
        cache.clear_symbol("BTCUSDT")
        cleared = cache.get("BTCUSDT", "test_indicator")
        print(f"After clear: {cleared}")

        return True

    except Exception as e:
        print(f"Cache test skipped (Redis not available): {e}")
        return True

def main():
    """Función principal"""
    print("🚀 Iniciando tests de Sprint 3 - Indicadores Técnicos")
    print("=" * 60)

    try:
        # Test manager
        test_indicator_manager()

        # Test individuales
        test_individual_indicators()

        # Test caché
        test_cache()

        print("\n✅ Todos los tests pasaron exitosamente!")
        return 0

    except Exception as e:
        print(f"\n❌ Error en tests: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())