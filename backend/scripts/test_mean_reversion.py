import asyncio
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.mean_reversion import MeanReversionStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.core.strategies.strategy_config import StrategyConfig
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test_mean_reversion")

async def main():
    """Test Mean Reversion Strategy"""

    logger.info("=" * 80)
    logger.info("MEAN REVERSION STRATEGY TEST")
    logger.info("=" * 80)

    symbol = "BTCUSDT"

    # Crear componentes
    indicator_manager = IndicatorManager()
    position_sizer = PositionSizer(account_balance=10000)
    signal_validator = SignalValidator(indicator_manager=indicator_manager)

    # Crear estrategia con configuración balanceada
    config = StrategyConfig.get_strategy_config("mean_reversion", "balanced")

    strategy = MeanReversionStrategy(
        symbol=symbol,
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator,
        **config
    )

    logger.info("Strategy created:")
    logger.info(f"  Name: {strategy.name}")
    logger.info(f"  Symbol: {strategy.symbol}")
    logger.info(f"  RSI Oversold: {strategy.rsi_oversold}")
    logger.info(f"  RSI Overbought: {strategy.rsi_overbought}")
    logger.info(f"  Volume Multiplier: {strategy.volume_multiplier}")

    # Test Case 1: Oversold condition (should trigger LONG)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 1: Oversold Reversal (LONG)")
    logger.info("=" * 60)

    # Setup indicators para condición oversold
    # Inicializar indicadores para el símbolo
    indicator_manager.initialize_symbol(symbol)
    
    # Agregar barras de datos históricos
    bars = [
        {
            'timestamp': 1000 + i,
            'open': 62000 - (i * 100),
            'high': 62100 - (i * 100),
            'low': 61900 - (i * 100),
            'close': 62000 - (i * 100),
            'volume': 1000 + (i * 50)
        }
        for i in range(30)
    ]
    
    # Actualizar indicadores con cada barra
    for bar in bars:
        indicator_manager.update_with_bar(symbol, bar)

    # Última barra: precio en lower band, RSI oversold, empieza a revertir
    last_bar = {
        'timestamp': 2000,
        'open': 59000,
        'high': 59100,
        'low': 58900,
        'close': 59050,  # Close > Open (reversión)
        'volume': 2000,  # Alto volumen
        'previous_close': 59000
    }

    # Simular condición oversold
    # Configurar indicadores manualmente para el test
    if 'indicators' not in indicator_manager.__dict__:
        indicator_manager.indicators = {}
    if symbol not in indicator_manager.indicators:
        indicator_manager.indicators[symbol] = {}
    
    # Simular valores calculados usando objetos reales
    # Para RSI
    if hasattr(indicator_manager.indicators[symbol]['rsi_14'], 'last_value'):
        indicator_manager.indicators[symbol]['rsi_14'].last_value = 28.0  # Oversold
    
    # Para ATR
    if hasattr(indicator_manager.indicators[symbol]['atr'], 'last_value'):
        indicator_manager.indicators[symbol]['atr'].last_value = 200
    
    # Para BB - ya están calculados por las barras anteriores
    # Pero podemos forzar los valores si es necesario
    if hasattr(indicator_manager.indicators[symbol]['bb'], 'upper'):
        indicator_manager.indicators[symbol]['bb'].upper = 63000
        indicator_manager.indicators[symbol]['bb'].middle = 61000
        indicator_manager.indicators[symbol]['bb'].lower = 59000

    # Debug: ver qué valores están disponibles
    all_values = indicator_manager.get_all_values(symbol)
    logger.info(f"Available indicators: {list(all_values.keys())}")
    logger.info(f"RSI value: {all_values.get('rsi_14')}")
    logger.info(f"BB values: upper={all_values.get('bb_upper')}, middle={all_values.get('bb_middle')}, lower={all_values.get('bb_lower')}")

    # Check setup
    signal = strategy.check_setup(last_bar)

    if signal:
        logger.info("✅ SIGNAL GENERATED:")
        logger.info(f"  Action: {signal.action}")
        logger.info(f"  Entry: ${signal.entry_price}")
        logger.info(f"  Stop Loss: ${signal.stop_loss}")
        logger.info(f"  Take Profit: ${signal.take_profit}")
        logger.info(f"  Quantity: {signal.quantity}")
        logger.info(f"  Confidence: {signal.confidence}")
        logger.info(f"  Setup Type: {signal.signal_type}")
    else:
        logger.warning("❌ No signal generated")

    # Test Case 2: Overbought condition (should trigger SHORT)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 2: Overbought Reversal (SHORT)")
    logger.info("=" * 60)

    # Última barra: precio en upper band, RSI overbought, empieza a revertir
    last_bar = {
        'timestamp': 2000,
        'open': 65000,
        'high': 65100,
        'low': 64900,
        'close': 64950,  # Close < Open (reversión)
        'volume': 2000,
        'previous_close': 65000
    }

    # Simular condición overbought
    if hasattr(indicator_manager.indicators[symbol]['rsi_14'], 'last_value'):
        indicator_manager.indicators[symbol]['rsi_14'].last_value = 72.0  # Overbought
    
    # Para BB - ya están calculados por las barras anteriores
    # Pero podemos forzar los valores si es necesario
    if hasattr(indicator_manager.indicators[symbol]['bb'], 'upper'):
        indicator_manager.indicators[symbol]['bb'].upper = 65000
        indicator_manager.indicators[symbol]['bb'].middle = 63000
        indicator_manager.indicators[symbol]['bb'].lower = 61000

    # Check setup
    signal = strategy.check_setup(last_bar)

    if signal:
        logger.info("✅ SIGNAL GENERATED:")
        logger.info(f"  Action: {signal.action}")
        logger.info(f"  Entry: ${signal.entry_price}")
        logger.info(f"  Stop Loss: ${signal.stop_loss}")
        logger.info(f"  Take Profit: ${signal.take_profit}")
        logger.info(f"  Quantity: {signal.quantity}")
        logger.info(f"  Confidence: {signal.confidence}")
        logger.info(f"  Setup Type: {signal.signal_type}")
    else:
        logger.warning("❌ No signal generated")

    # Test Case 3: No setup (neutral RSI)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 3: Neutral Condition (No Signal)")
    logger.info("=" * 60)

    # RSI neutral
    if hasattr(indicator_manager.indicators[symbol]['rsi_14'], 'last_value'):
        indicator_manager.indicators[symbol]['rsi_14'].last_value = 50.0  # Neutral

    # Usar una barra neutral (close = open, sin dirección clara)
    neutral_bar = {
        'timestamp': 2000,
        'open': 62000,
        'high': 62100,
        'low': 61900,
        'close': 62000,  # Close = Open (neutral)
        'volume': 1500,
        'previous_close': 62000
    }

    signal = strategy.check_setup(neutral_bar)

    if signal:
        logger.warning(f"❌ Unexpected signal: {signal.action}")
    else:
        logger.info("✅ Correctly no signal (RSI neutral)")

    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL TESTS COMPLETED")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())