from typing import Dict, List, Optional
from app.core.indicators.atr import ATR
from app.core.indicators.bollinger import BollingerBands
from app.core.indicators.rsi import RSI
from app.core.indicators.macd import MACD
from app.core.indicators.adx import ADX
from app.core.indicators.vwap import VWAP
from app.core.indicators.obi import OBI
from app.core.indicators.cache import IndicatorCache
from app.utils.logger import get_logger
from datetime import datetime

logger = get_logger("indicator_manager")

class IndicatorManager:
    """
    Manager central de indicadores
    - Crea y mantiene indicadores por símbolo
    - Actualiza con nuevas barras
    - Caché en Redis
    """

    def __init__(self):
        # Indicadores por símbolo
        # estructura: {symbol: {indicator_name: instance}}
        self.indicators: Dict[str, Dict[str, any]] = {}

        # Caché
        self.cache = IndicatorCache()

        # Configuraciones por defecto
        self.default_configs = {
            'atr': {'period': 14},
            'bb': {'period': 20, 'num_std': 2.0},
            'adx': {'period': 14},
            'rsi_2': {'period': 2},
            'rsi_3': {'period': 3},
            'rsi_5': {'period': 5},
            'rsi_7': {'period': 7},
            'rsi_14': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},  # Más rápido para scalping
            'vwap': {}
        }

    def initialize_symbol(self, symbol: str):
        """Inicializar todos los indicadores para un símbolo"""
        if symbol in self.indicators:
            logger.warning("symbol_already_initialized", symbol=symbol)
            return

        self.indicators[symbol] = {}

        # Crear instancias de indicadores
        self.indicators[symbol]['atr'] = ATR(symbol, **self.default_configs['atr'])
        self.indicators[symbol]['bb'] = BollingerBands(symbol, **self.default_configs['bb'])
        self.indicators[symbol]['adx'] = ADX(symbol, **self.default_configs['adx'])
        self.indicators[symbol]['rsi_2'] = RSI(symbol, period=2)
        self.indicators[symbol]['rsi_3'] = RSI(symbol, period=3)
        self.indicators[symbol]['rsi_5'] = RSI(symbol, period=5)
        self.indicators[symbol]['rsi_7'] = RSI(symbol, period=7)
        self.indicators[symbol]['rsi_14'] = RSI(symbol, period=14)
        self.indicators[symbol]['macd'] = MACD(symbol, **self.default_configs['macd'])
        self.indicators[symbol]['vwap'] = VWAP(symbol)
        self.indicators[symbol]['obi'] = OBI(symbol)

        logger.info("indicators_initialized", symbol=symbol, count=len(self.indicators[symbol]))

    def update_with_bar(self, symbol: str, bar: Dict):
        """
        Actualizar todos los indicadores con nueva barra

        Args:
            symbol: Símbolo (ej: BTCUSDT)
            bar: Dict con keys: timestamp, open, high, low, close, volume
        """
        if symbol not in self.indicators:
            self.initialize_symbol(symbol)

        indicators = self.indicators[symbol]
        results = {}

        try:
            # Actualizar cada indicador
            for name, indicator in indicators.items():
                if name == 'obi':
                    # OBI se actualiza con orderbook, no con barras
                    continue

                value = indicator.update(bar)

                if value is not None:
                    results[name] = value

                    # Cachear si está listo
                    if indicator.is_ready:
                        self.cache.set(
                            symbol,
                            name,
                            indicator.get_state(),
                            ttl=60  # 1 minuto
                        )

            logger.debug(
                "indicators_updated",
                symbol=symbol,
                bar_time=bar.get('timestamp'),
                ready_count=len(results)
            )

            return results

        except Exception as e:
            logger.error("update_indicators_error", symbol=symbol, error=str(e))
            return {}

    def update_with_orderbook(self, symbol: str, orderbook: Dict):
        """
        Actualizar OBI con orderbook snapshot

        Args:
            orderbook: Dict con keys: bids, asks (listas de tuplas)
        """
        if symbol not in self.indicators:
            self.initialize_symbol(symbol)

        try:
            obi = self.indicators[symbol]['obi']

            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            # Calcular OBI para diferentes profundidades
            obi_5 = obi.calculate(bids, asks, depth=5)
            obi_10 = obi.calculate(bids, asks, depth=10)

            # Cachear
            self.cache.set(
                symbol,
                'obi',
                {
                    'obi_5': obi_5,
                    'obi_10': obi_10,
                    'last_update': str(datetime.utcnow())
                },
                ttl=10  # 10 segundos (orderbook cambia rápido)
            )

            return {'obi_5': obi_5, 'obi_10': obi_10}

        except Exception as e:
            logger.error("update_obi_error", symbol=symbol, error=str(e))
            return {}

    def get_indicator(self, symbol: str, indicator_name: str):
        """Obtener instancia de un indicador específico"""
        if symbol not in self.indicators:
            return None
        return self.indicators[symbol].get(indicator_name)

    def get_all_values(self, symbol: str) -> Dict:
        """
        Obtener valores actuales de todos los indicadores

        Returns:
            Dict con todos los valores de indicadores
        """
        if symbol not in self.indicators:
            return {}

        values = {}

        for name, indicator in self.indicators[symbol].items():
            if name == 'obi':
                # OBI tiene estructura especial
                values['obi'] = indicator.get_state()
            elif hasattr(indicator, 'last_value'):
                values[name] = indicator.last_value

                # Para indicadores complejos, añadir detalles
                if name == 'bb' and indicator.upper is not None:
                    values['bb_upper'] = indicator.upper
                    values['bb_middle'] = indicator.middle
                    values['bb_lower'] = indicator.lower
                    values['bb_bandwidth'] = indicator.bandwidth

                elif name == 'macd' and indicator.signal_line is not None:
                    values['macd_line'] = indicator.macd_line
                    values['macd_signal'] = indicator.signal_line
                    values['macd_histogram'] = indicator.histogram
                elif name == 'adx' and indicator.last_value is not None:
                    values['adx'] = indicator.last_value
                    values['plus_di'] = indicator.plus_di
                    values['minus_di'] = indicator.minus_di

        return values

    def is_ready(self, symbol: str, required_indicators: List[str] = None) -> bool:
        """
        Verificar si indicadores están listos para operar

        Args:
            symbol: Símbolo
            required_indicators: Lista de indicadores requeridos, o None para todos
        """
        if symbol not in self.indicators:
            return False

        indicators_to_check = required_indicators or list(self.indicators[symbol].keys())

        for name in indicators_to_check:
            if name == 'obi':
                continue  # OBI siempre está "ready"

            indicator = self.indicators[symbol].get(name)
            if indicator and not indicator.is_ready:
                return False

        return True

    def reset_symbol(self, symbol: str):
        """Resetear todos los indicadores de un símbolo"""
        if symbol in self.indicators:
            for indicator in self.indicators[symbol].values():
                if hasattr(indicator, 'reset'):
                    indicator.reset()

            # Limpiar caché
            self.cache.clear_symbol(symbol)

            logger.info("indicators_reset", symbol=symbol)

    def get_signals(self, symbol: str) -> Dict:
        """
        Generar señales de trading basadas en indicadores

        Returns:
            Dict con señales: {'bb_breakout': bool, 'rsi_oversold': bool, ...}
        """
        if symbol not in self.indicators:
            return {}

        ind = self.indicators[symbol]
        signals = {}

        try:
            # Señal de Bollinger Bands: Squeeze (compresión)
            bb = ind.get('bb')
            if bb and bb.is_ready:
                signals['bb_compressed'] = bb.is_compressed(threshold=0.02)
                signals['bb_bandwidth'] = bb.bandwidth

            # Señales de RSI: Extremos
            rsi_2 = ind.get('rsi_2')
            if rsi_2 and rsi_2.is_ready:
                signals['rsi_2_oversold'] = rsi_2.is_oversold(threshold=10)
                signals['rsi_2_overbought'] = rsi_2.is_overbought(threshold=90)

            rsi_3 = ind.get('rsi_3')
            if rsi_3 and rsi_3.is_ready:
                signals['rsi_3_extreme'] = rsi_3.is_extreme(overbought=80, oversold=20)

            # Señal de MACD: Cruces
            macd = ind.get('macd')
            if macd and macd.signal_line is not None:
                signals['macd_bullish_cross'] = macd.is_bullish_crossover()
                signals['macd_bearish_cross'] = macd.is_bearish_crossover()

            # Señal de OBI: Presión
            obi = ind.get('obi')
            if obi:
                signals['obi_buy_pressure'] = obi.is_buy_pressure(threshold=0.62)
                signals['obi_sell_pressure'] = obi.is_sell_pressure(threshold=0.38)
                signals['obi_trend'] = obi.get_trend(window=10)

            return signals

        except Exception as e:
            logger.error("get_signals_error", symbol=symbol, error=str(e))
            return {}