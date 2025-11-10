import asyncio
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.mean_reversion import MeanReversionStrategy
from app.core.strategies.signal_aggregator import SignalAggregator
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.core.strategies.strategy_config import StrategyConfig
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test_integration")

async def main():
    """Integration Test: Mean Reversion + Signal Aggregator"""

    logger.info("=" * 80)
    logger.info("INTEGRATION TEST: MEAN REVERSION + SIGNAL AGGREGATOR")
    logger.info("=" * 80)

    symbol = "BTCUSDT"

    # Crear componentes compartidos
    indicator_manager = IndicatorManager()
    position_sizer = PositionSizer(account_balance=10000)
    signal_validator = SignalValidator(indicator_manager=indicator_manager)
    signal_aggregator = SignalAggregator()

    # Crear estrategia Mean Reversion
    mr_config = StrategyConfig.get_strategy_config("mean_reversion", "balanced")
    mean_reversion = MeanReversionStrategy(
        symbol=symbol,
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator,
        **mr_config
    )

    logger.info("Strategy created:")
    logger.info(f"  Mean Reversion: {mean_reversion.name}")

    # Inicializar indicadores
    indicator_manager.initialize_symbol(symbol)

    # Agregar barras de datos históricos
    bars = [
        {
            'timestamp': 1000 + i,
            'open': 62000 - (i * 50),
            'high': 62100 - (i * 50),
            'low': 61900 - (i * 50),
            'close': 62000 - (i * 50),
            'volume': 1000 + (i * 100)
        }
        for i in range(50)  # Más barras para mejores indicadores
    ]

    # Actualizar indicadores con cada barra
    for bar in bars:
        indicator_manager.update_with_bar(symbol, bar)

    # Test Case: Oversold condition where both strategies might agree
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE: Oversold Condition - Potential Agreement")
    logger.info("=" * 60)

    # Configurar condición oversold para Mean Reversion
    if hasattr(indicator_manager.indicators[symbol]['rsi_14'], 'last_value'):
        indicator_manager.indicators[symbol]['rsi_14'].last_value = 25.0  # Very oversold

    if hasattr(indicator_manager.indicators[symbol]['bb'], 'upper'):
        indicator_manager.indicators[symbol]['bb'].upper = 62500
        indicator_manager.indicators[symbol]['bb'].middle = 61500
        indicator_manager.indicators[symbol]['bb'].lower = 60500

    if hasattr(indicator_manager.indicators[symbol]['atr'], 'last_value'):
        indicator_manager.indicators[symbol]['atr'].last_value = 200

    # Barra de test: precio cerca de lower BB, close > open (reversión)
    test_bar = {
        'timestamp': 3000,
        'open': 60400,
        'high': 60500,
        'low': 60300,
        'close': 60450,  # Close > Open (reversión alcista)
        'volume': 2500,
        'previous_close': 60400
    }

    # Para este test, solo usaremos Mean Reversion
    # El Signal Aggregator ya está probado por separado

    # Generar señal de Mean Reversion
    logger.info("Generating signal from Mean Reversion strategy...")

    mr_signal = mean_reversion.check_setup(test_bar)
    if mr_signal:
        logger.info(f"✅ Mean Reversion signal: {mr_signal.action} (confidence: {mr_signal.confidence})")
        signal_aggregator.add_signal(mr_signal)

        # Obtener señal agregada (debería ser la misma ya que solo hay una)
        aggregated_signal = signal_aggregator.get_aggregated_signal(symbol)

        if aggregated_signal:
            logger.info("✅ AGGREGATED SIGNAL:")
            logger.info(f"  Action: {aggregated_signal.action}")
            logger.info(f"  Entry Price: ${aggregated_signal.entry_price}")
            logger.info(f"  Confidence: {aggregated_signal.confidence}")
            logger.info(f"  Source Strategies: {aggregated_signal.source_signals}")
        else:
            logger.info("❌ No aggregated signal")
    else:
        logger.info("❌ No Mean Reversion signal")

    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST COMPLETED")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())