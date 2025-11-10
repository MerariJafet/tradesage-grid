import pytest
import asyncio
from datetime import datetime, timedelta
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.indicators.cache import IndicatorCache
from app.core.indicators.atr import ATR
from app.core.indicators.bollinger import BollingerBands
from app.core.indicators.rsi import RSI
from app.core.indicators.macd import MACD
from app.core.indicators.vwap import VWAP
from app.core.indicators.obi import OBI

class TestIndicatorManager:
    """Tests para IndicatorManager"""

    @pytest.fixture
    def manager(self):
        return IndicatorManager()

    def test_initialization(self, manager):
        """Test inicialización del manager"""
        assert len(manager.indicators) == 0
        assert manager.cache is not None

    def test_symbol_initialization(self, manager):
        """Test inicialización de indicadores para un símbolo"""
        symbol = "BTCUSDT"
        manager.initialize_symbol(symbol)

        assert symbol in manager.indicators
        assert 'atr' in manager.indicators[symbol]
        assert 'bb' in manager.indicators[symbol]
        assert 'rsi_2' in manager.indicators[symbol]
        assert 'macd' in manager.indicators[symbol]
        assert 'vwap' in manager.indicators[symbol]
        assert 'obi' in manager.indicators[symbol]

    def test_update_with_bar(self, manager):
        """Test actualización con barras"""
        symbol = "BTCUSDT"

        # Barra de ejemplo
        bar = {
            'timestamp': datetime.utcnow(),
            'open': 50000,
            'high': 51000,
            'low': 49000,
            'close': 50500,
            'volume': 100
        }

        results = manager.update_with_bar(symbol, bar)

        # Debería tener algunos resultados (dependiendo del estado)
        assert isinstance(results, dict)

        # Verificar que se inicializó el símbolo
        assert symbol in manager.indicators

    def test_update_with_orderbook(self, manager):
        """Test actualización con orderbook"""
        symbol = "BTCUSDT"

        # Orderbook de ejemplo
        orderbook = {
            'bids': [(50000, 1.0), (49900, 2.0), (49800, 3.0)],
            'asks': [(50100, 1.0), (50200, 2.0), (50300, 3.0)]
        }

        results = manager.update_with_orderbook(symbol, orderbook)

        assert 'obi_5' in results
        assert 'obi_10' in results
        assert isinstance(results['obi_5'], float)
        assert isinstance(results['obi_10'], float)

    def test_get_all_values(self, manager):
        """Test obtener todos los valores"""
        symbol = "BTCUSDT"

        # Inicializar y actualizar con algunas barras
        manager.initialize_symbol(symbol)

        # Añadir varias barras para que los indicadores se "calienten"
        base_time = datetime.utcnow()
        for i in range(30):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 50000 + i * 100,
                'high': 51000 + i * 100,
                'low': 49000 + i * 100,
                'close': 50500 + i * 100,
                'volume': 100 + i
            }
            manager.update_with_bar(symbol, bar)

        values = manager.get_all_values(symbol)

        assert isinstance(values, dict)
        assert 'atr' in values
        assert 'bb_upper' in values
        assert 'rsi_2' in values

    def test_is_ready(self, manager):
        """Test verificación de readiness"""
        symbol = "BTCUSDT"

        # No debería estar listo inicialmente
        assert not manager.is_ready(symbol)

        # Después de inicializar
        manager.initialize_symbol(symbol)
        assert not manager.is_ready(symbol)  # Aún no tiene suficientes datos

        # Después de suficientes barras
        base_time = datetime.utcnow()
        for i in range(30):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 50000 + i * 10,
                'high': 51000 + i * 10,
                'low': 49000 + i * 10,
                'close': 50500 + i * 10,
                'volume': 100
            }
            manager.update_with_bar(symbol, bar)

        # Ahora debería estar listo para algunos indicadores
        assert manager.is_ready(symbol, ['rsi_2'])  # RSI necesita menos datos

    def test_get_signals(self, manager):
        """Test generación de señales"""
        symbol = "BTCUSDT"

        # Inicializar y alimentar con datos
        manager.initialize_symbol(symbol)

        base_time = datetime.utcnow()
        for i in range(50):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 50000 + i * 50,
                'high': 51000 + i * 50,
                'low': 49000 + i * 50,
                'close': 50500 + i * 50,
                'volume': 100
            }
            manager.update_with_bar(symbol, bar)

        signals = manager.get_signals(symbol)

        assert isinstance(signals, dict)
        assert 'bb_compressed' in signals
        assert 'rsi_2_oversold' in signals
        assert 'macd_bullish_cross' in signals

    def test_reset_symbol(self, manager):
        """Test reset de indicadores"""
        symbol = "BTCUSDT"

        # Inicializar y actualizar
        manager.initialize_symbol(symbol)

        bar = {
            'timestamp': datetime.utcnow(),
            'open': 50000,
            'high': 51000,
            'low': 49000,
            'close': 50500,
            'volume': 100
        }
        manager.update_with_bar(symbol, bar)

        # Reset
        manager.reset_symbol(symbol)

        # Verificar que se reseteó (valores deberían ser None o iniciales)
        values = manager.get_all_values(symbol)
        # Los indicadores deberían estar en estado inicial

class TestATR:
    """Tests para ATR"""

    def test_atr_calculation(self):
        atr = ATR("BTCUSDT", period=14)

        # Primera barra
        bar1 = {'high': 51000, 'low': 49000, 'close': 50000}
        atr.update(bar1)
        assert atr.last_value is None  # No hay suficiente data

        # Más barras
        for i in range(14):
            bar = {
                'high': 50000 + i * 100,
                'low': 49000 + i * 100,
                'close': 49500 + i * 100
            }
            atr.update(bar)

        assert atr.last_value is not None
        assert atr.is_ready

    def test_atr_multiple(self):
        atr = ATR("BTCUSDT", period=14)

        # Alimentar con suficientes datos
        for i in range(20):
            bar = {
                'high': 50000 + i * 50,
                'low': 49000 + i * 50,
                'close': 49500 + i * 50
            }
            atr.update(bar)

        multiple = atr.get_atr_multiple(50500, 51000)
        assert isinstance(multiple, float)
        assert multiple >= 0

class TestBollingerBands:
    """Tests para Bollinger Bands"""

    def test_bb_calculation(self):
        bb = BollingerBands("BTCUSDT", period=20, num_std=2.0)

        # Alimentar con datos
        for i in range(25):
            bar = {'close': 50000 + i * 10}
            bb.update(bar)

        assert bb.is_ready
        assert bb.upper > bb.middle > bb.lower
        assert bb.bandwidth > 0

    def test_bb_compression(self):
        bb = BollingerBands("BTCUSDT", period=20, num_std=2.0)

        # Datos con poca volatilidad (compresión)
        for i in range(25):
            bar = {'close': 50000 + (i % 3)}  # Muy poca variación
            bb.update(bar)

        assert bb.is_compressed(threshold=0.01)  # Debería estar comprimido

    def test_bb_position(self):
        bb = BollingerBands("BTCUSDT", period=20, num_std=2.0)

        # Alimentar con datos
        for i in range(25):
            bar = {'close': 50000 + i * 10}
            bb.update(bar)

        position = bb.get_position(51000)  # Precio arriba de la banda superior
        assert position > 1.0

class TestRSI:
    """Tests para RSI"""

    def test_rsi_calculation(self):
        rsi = RSI("BTCUSDT", period=14)

        # Alimentar con datos ascendentes (debería dar RSI alto)
        for i in range(20):
            bar = {'close': 50000 + i * 100}
            rsi.update(bar)

        assert rsi.is_ready
        assert rsi.last_value > 70  # RSI alto en tendencia alcista

    def test_rsi_oversold_overbought(self):
        rsi = RSI("BTCUSDT", period=2)

        # Oversold: precios bajando
        for i in range(5):
            bar = {'close': 50000 - i * 1000}
            rsi.update(bar)

        assert rsi.is_oversold(threshold=30)

        # Reset para overbought
        rsi.reset()

        # Overbought: precios subiendo
        for i in range(5):
            bar = {'close': 50000 + i * 1000}
            rsi.update(bar)

        assert rsi.is_overbought(threshold=70)

class TestMACD:
    """Tests para MACD"""

    def test_macd_calculation(self):
        macd = MACD("BTCUSDT", fast=12, slow=26, signal=9)

        # Alimentar con suficientes datos
        for i in range(40):
            bar = {'close': 50000 + i * 50}
            macd.update(bar)

        assert macd.is_ready
        assert macd.macd_line is not None
        assert macd.signal_line is not None
        assert macd.histogram is not None

    def test_macd_crossover(self):
        macd = MACD("BTCUSDT", fast=12, slow=26, signal=9)

        # Tendencia bajista primero
        for i in range(20):
            bar = {'close': 50000 - i * 100}
            macd.update(bar)

        # Luego alcista
        for i in range(20):
            bar = {'close': 50000 - 2000 + i * 200}
            macd.update(bar)

        # Debería haber un cruce alcista
        assert macd.is_bullish_crossover()

class TestVWAP:
    """Tests para VWAP"""

    def test_vwap_calculation(self):
        vwap = VWAP("BTCUSDT")

        # Múltiples barras en el mismo día
        base_time = datetime.utcnow().replace(hour=9, minute=0, second=0)

        for i in range(10):
            bar = {
                'timestamp': base_time + timedelta(minutes=i*5),
                'close': 50000 + i * 100,
                'volume': 100 + i * 10
            }
            vwap.update(bar)

        assert vwap.is_ready
        assert vwap.last_value > 0

    def test_vwap_deviation(self):
        vwap = VWAP("BTCUSDT")

        # Alimentar datos
        base_time = datetime.utcnow().replace(hour=9, minute=0, second=0)

        for i in range(10):
            bar = {
                'timestamp': base_time + timedelta(minutes=i*5),
                'close': 50000 + i * 100,
                'volume': 100
            }
            vwap.update(bar)

        deviation = vwap.get_deviation_pct(51000)
        assert isinstance(deviation, float)

class TestOBI:
    """Tests para Order Book Imbalance"""

    def test_obi_calculation(self):
        obi = OBI("BTCUSDT")

        # Orderbook desbalanceado al alza
        bids = [(50000, 10.0), (49900, 8.0), (49800, 6.0)]
        asks = [(50100, 1.0), (50200, 1.0), (50300, 1.0)]

        obi_value = obi.calculate(bids, asks, depth=5)
        assert obi_value > 0.5  # Debería estar desbalanceado al alza

    def test_obi_buy_pressure(self):
        obi = OBI("BTCUSDT")

        # Presión de compra
        bids = [(50000, 10.0), (49900, 8.0)]
        asks = [(50100, 1.0), (50200, 1.0)]

        obi.calculate(bids, asks, depth=5)
        assert obi.is_buy_pressure(threshold=0.6)

    def test_obi_sell_pressure(self):
        obi = OBI("BTCUSDT")

        # Presión de venta
        bids = [(50000, 1.0), (49900, 1.0)]
        asks = [(50100, 10.0), (50200, 8.0)]

        obi.calculate(bids, asks, depth=5)
        assert obi.is_sell_pressure(threshold=0.4)

class TestIndicatorCache:
    """Tests para IndicatorCache"""

    def test_cache_operations(self):
        cache = IndicatorCache()

        # Set y get
        cache.set("BTCUSDT", "rsi", {"value": 70.5}, ttl=60)
        cached = cache.get("BTCUSDT", "rsi")

        assert cached["value"] == 70.5

    def test_cache_expiration(self):
        cache = IndicatorCache()

        # Set con TTL corto
        cache.set("BTCUSDT", "atr", {"value": 1000}, ttl=1)

        # Esperar expiración
        import time
        time.sleep(2)

        cached = cache.get("BTCUSDT", "atr")
        assert cached is None

    def test_clear_symbol(self):
        cache = IndicatorCache()

        # Set múltiples indicadores
        cache.set("BTCUSDT", "rsi", {"value": 70}, ttl=60)
        cache.set("BTCUSDT", "atr", {"value": 1000}, ttl=60)

        # Clear símbolo
        cache.clear_symbol("BTCUSDT")

        assert cache.get("BTCUSDT", "rsi") is None
        assert cache.get("BTCUSDT", "atr") is None

# Tests de integración
class TestIntegration:
    """Tests de integración end-to-end"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test pipeline completo: WS -> Indicadores -> Señales"""
        manager = IndicatorManager()

        # Simular recepción de barras como vendría del WS
        symbol = "BTCUSDT"
        base_time = datetime.utcnow()

        # Generar datos realistas (tendencia alcista)
        bars = []
        for i in range(100):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 50000 + i * 20,
                'high': 50200 + i * 20,
                'low': 49800 + i * 20,
                'close': 50100 + i * 20,
                'volume': 100 + i * 5
            }
            bars.append(bar)

        # Procesar barras
        for bar in bars:
            manager.update_with_bar(symbol, bar)

        # Verificar indicadores
        values = manager.get_all_values(symbol)
        assert len(values) > 0

        # Verificar señales
        signals = manager.get_signals(symbol)
        assert len(signals) > 0

        # Verificar readiness
        assert manager.is_ready(symbol, ['rsi_2', 'bb'])

        # Verificar caché
        cached = manager.cache.get_all_symbol(symbol)
        assert len(cached) > 0