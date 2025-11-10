import asyncio
from app.core.strategies.signal_aggregator import SignalAggregator
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test_signal_aggregator")

async def main():
    """Test Signal Aggregator"""

    logger.info("=" * 80)
    logger.info("SIGNAL AGGREGATOR TEST")
    logger.info("=" * 80)

    aggregator = SignalAggregator()
    symbol = "BTCUSDT"

    logger.info("Signal Aggregator created")
    logger.info(f"  Strategy weights: {aggregator.strategy_weights}")
    logger.info(f"  Min confidence threshold: {aggregator.min_confidence_threshold}")

    # Test Case 1: Single signal (should pass through)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 1: Single Signal Pass-Through")
    logger.info("=" * 60)

    signal1 = TradingSignal(
        strategy_name="BreakoutCompression",
        symbol=symbol,
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62100.0,
        stop_loss=62000.0,
        take_profit=62250.0,
        quantity=0.016,
        confidence=0.8,
        reason="Strong breakout above upper BB"
    )

    aggregator.add_signal(signal1)
    aggregated = aggregator.get_aggregated_signal(symbol)

    if aggregated:
        logger.info("✅ AGGREGATED SIGNAL:")
        logger.info(f"  Action: {aggregated.action}")
        logger.info(f"  Entry: ${aggregated.entry_price}")
        logger.info(f"  Confidence: {aggregated.confidence}")
        logger.info(f"  Strategy: {aggregated.strategy_name}")
        logger.info(f"  Source signals: {aggregated.source_signals}")
    else:
        logger.warning("❌ No aggregated signal")

    # Test Case 2: Conflicting signals (Breakout BUY vs Mean Reversion SELL)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 2: Conflicting Signals")
    logger.info("=" * 60)

    # Clear previous signals
    aggregator.clear_signals(symbol)

    # Breakout BUY (high confidence)
    signal2_buy = TradingSignal(
        strategy_name="BreakoutCompression",
        symbol=symbol,
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62100.0,
        stop_loss=62000.0,
        take_profit=62250.0,
        quantity=0.016,
        confidence=0.9,
        reason="Strong breakout"
    )

    # Mean Reversion SELL (lower confidence)
    signal2_sell = TradingSignal(
        strategy_name="MeanReversion",
        symbol=symbol,
        signal_type=SignalType.MEAN_REVERSION,
        action=SignalAction.SELL,
        entry_price=62050.0,
        stop_loss=62100.0,
        take_profit=61900.0,
        quantity=0.015,
        confidence=0.6,
        reason="Overbought reversal"
    )

    aggregator.add_signal(signal2_buy)
    aggregator.add_signal(signal2_sell)

    summary = aggregator.get_signal_summary(symbol)
    logger.info(f"Signal summary: {summary}")

    aggregated = aggregator.get_aggregated_signal(symbol)

    if aggregated:
        logger.info("✅ AGGREGATED SIGNAL:")
        logger.info(f"  Action: {aggregated.action}")
        logger.info(f"  Entry: ${aggregated.entry_price}")
        logger.info(f"  Confidence: {aggregated.confidence}")
        logger.info(f"  Source signals: {aggregated.source_signals}")
    else:
        logger.warning("❌ No aggregated signal")

    # Test Case 3: Reinforcing signals (both BUY)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 3: Reinforcing Signals")
    logger.info("=" * 60)

    # Clear previous signals
    aggregator.clear_signals(symbol)

    # Breakout BUY
    signal3_breakout = TradingSignal(
        strategy_name="BreakoutCompression",
        symbol=symbol,
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62100.0,
        stop_loss=62000.0,
        take_profit=62250.0,
        quantity=0.016,
        confidence=0.7,
        reason="Breakout above upper BB"
    )

    # Mean Reversion BUY (oversold)
    signal3_mr = TradingSignal(
        strategy_name="MeanReversion",
        symbol=symbol,
        signal_type=SignalType.MEAN_REVERSION,
        action=SignalAction.BUY,
        entry_price=62080.0,
        stop_loss=61980.0,
        take_profit=62200.0,
        quantity=0.017,
        confidence=0.8,
        reason="Oversold reversal"
    )

    aggregator.add_signal(signal3_breakout)
    aggregator.add_signal(signal3_mr)

    summary = aggregator.get_signal_summary(symbol)
    logger.info(f"Signal summary: {summary}")

    aggregated = aggregator.get_aggregated_signal(symbol)

    if aggregated:
        logger.info("✅ AGGREGATED SIGNAL:")
        logger.info(f"  Action: {aggregated.action}")
        logger.info(f"  Entry: ${aggregated.entry_price}")
        logger.info(f"  Confidence: {aggregated.confidence}")
        logger.info(f"  Source signals: {aggregated.source_signals}")
    else:
        logger.warning("❌ No aggregated signal")

    # Test Case 4: Low confidence signals (should be filtered)
    logger.info("\n" + "=" * 60)
    logger.info("TEST CASE 4: Low Confidence Signals")
    logger.info("=" * 60)

    # Clear previous signals
    aggregator.clear_signals(symbol)

    # Low confidence signal
    signal4_low = TradingSignal(
        strategy_name="BreakoutCompression",
        symbol=symbol,
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62100.0,
        stop_loss=62000.0,
        take_profit=62250.0,
        quantity=0.016,
        confidence=0.2,  # Below threshold
        reason="Weak breakout"
    )

    aggregator.add_signal(signal4_low)
    aggregated = aggregator.get_aggregated_signal(symbol)

    if aggregated:
        logger.warning(f"❌ Unexpected signal with low confidence: {aggregated.confidence}")
    else:
        logger.info("✅ Correctly filtered low confidence signal")

    logger.info("\n" + "=" * 80)
    logger.info("ALL SIGNAL AGGREGATOR TESTS COMPLETED")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())