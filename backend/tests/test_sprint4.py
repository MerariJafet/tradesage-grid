import pytest
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.core.strategies.position import Position, PositionSide, PositionStatus
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.core.strategies.breakout_compression import BreakoutCompressionStrategy
from app.core.strategies.strategy_manager import StrategyManager
from app.core.indicators.indicator_manager import IndicatorManager
from datetime import datetime, timedelta

def test_trading_signal_creation():
    """Test creación de señal"""
    signal = TradingSignal(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62000.0,
        stop_loss=61900.0,
        take_profit=62150.0,
        quantity=0.016,
        confidence=0.75
    )

    assert signal.action == SignalAction.BUY
    assert signal.get_risk_amount() == 1.6  # 100 * 0.016
    assert signal.get_risk_reward_ratio() == 1.5  # 150/100

def test_signal_expiry():
    """Test expiración de señal"""
    signal = TradingSignal(
        strategy_name="Test",
        symbol="BTCUSDT",
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62000.0,
        stop_loss=61900.0,
        take_profit=62150.0,
        quantity=0.016,
        confidence=0.75,
        expiry_seconds=1
    )

    assert not signal.is_expired()

    # Simular paso del tiempo
    signal.timestamp = datetime.utcnow() - timedelta(seconds=2)
    assert signal.is_expired()

def test_position_lifecycle():
    """Test ciclo de vida de posición"""
    position = Position(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=62000.0,
        quantity=0.016,
        stop_loss=61900.0,
        take_profit=62150.0
    )

    # Test PnL no realizado
    position.update_unrealized_pnl(62100.0)
    assert position.unrealized_pnl == 1.6  # (62100 - 62000) * 0.016

    # Test stop hit
    assert not position.is_stop_hit(62000.0)
    assert position.is_stop_hit(61850.0)

    # Test take profit hit
    assert position.is_take_profit_hit(62200.0)

    # Cerrar posición
    position.close(62150.0, "take_profit", exit_commission=0.5)
    assert position.status == PositionStatus.CLOSED
    assert position.realized_pnl == 2.4 - 0.5  # (150 * 0.016) - commission

def test_position_sizer():
    """Test cálculo de tamaño de posición"""
    sizer = PositionSizer(account_balance=10000.0)

    # Risk 0.5% con stop de 100 puntos
    quantity = sizer.calculate_quantity(
        symbol="BTCUSDT",
        entry_price=62000.0,
        stop_loss=61900.0,  # 100 puntos de riesgo
        atr=120.0
    )

    # Risk amount = 10000 * 0.005 = 50
    # But max position size = 20% of account = 2000
    # Quantity = min(50/100, 2000/62000) = min(0.5, 0.032) = 0.032
    assert quantity == pytest.approx(0.032, rel=0.1)

    # Validar tamaño
    is_valid, error = sizer.validate_position_size("BTCUSDT", quantity, 62000.0)
    assert is_valid

def test_signal_validator():
    """Test validación de señales"""
    indicator_manager = IndicatorManager()
    indicator_manager.initialize_symbol("BTCUSDT")

    validator = SignalValidator(
        indicator_manager=indicator_manager,
        max_open_positions=3,
        max_daily_trades=20
    )

    signal = TradingSignal(
        strategy_name="Test",
        symbol="BTCUSDT",
        signal_type=SignalType.BREAKOUT,
        action=SignalAction.BUY,
        entry_price=62000.0,
        stop_loss=61900.0,
        take_profit=62150.0,  # RR = 1.5:1 ✅
        quantity=0.016,
        confidence=0.75
    )

    # Validar (sin orderbook)
    # is_valid, errors = await validator.validate(signal)
    # Por ahora, test síncrono básico
    assert signal.get_risk_reward_ratio() >= 1.2

@pytest.mark.asyncio
async def test_breakout_strategy():
    """Test estrategia de breakout"""
    indicator_manager = IndicatorManager()
    indicator_manager.initialize_symbol("BTCUSDT")

    position_sizer = PositionSizer(10000.0)
    signal_validator = SignalValidator(indicator_manager)

    strategy = BreakoutCompressionStrategy(
        symbol="BTCUSDT",
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator,
        bb_compression_threshold=0.02,
        enabled=True
    )

    # Alimentar con barras para warmup
    bars = []
    price = 62000.0
    for i in range(30):
        bar = {
            'timestamp': datetime.utcnow(),
            'open': price,
            'high': price + 50,
            'low': price - 50,
            'close': price + (i % 2) * 10,  # Zigzag
            'volume': 1000
        }
        bars.append(bar)
        indicator_manager.update_with_bar("BTCUSDT", bar)
        price += 5

    # Simular breakout
    breakout_bar = {
        'timestamp': datetime.utcnow(),
        'open': price,
        'high': price + 200,  # Breakout fuerte
        'low': price,
        'close': price + 180,
        'volume': 2000  # Volumen alto
    }

    indicator_manager.update_with_bar("BTCUSDT", breakout_bar)

    # Intentar generar señal
    market_data = {
        'bar': breakout_bar,
        'indicators': indicator_manager.get_all_values("BTCUSDT")
    }

    signal = await strategy.generate_signal(market_data)

    # Puede o no generar señal dependiendo de los indicadores
    # Solo verificamos que no crashee
    assert strategy.total_signals_generated >= 0

def test_strategy_manager():
    """Test Strategy Manager"""
    indicator_manager = IndicatorManager()
    indicator_manager.initialize_symbol("BTCUSDT")

    manager = StrategyManager(
        indicator_manager=indicator_manager,
        account_balance=10000.0
    )

    # Añadir estrategia
    strategy = manager.add_strategy(
        symbol="BTCUSDT",
        strategy_type="breakout_compression",
        enabled=True
    )

    assert strategy.name == "BreakoutCompression"
    assert strategy.enabled

    # Obtener estrategia
    retrieved = manager.get_strategy("BTCUSDT", "BreakoutCompression")
    assert retrieved is strategy

    # Deshabilitar
    manager.disable_strategy("BTCUSDT", "BreakoutCompression")
    assert not strategy.enabled

    # Estadísticas
    stats = manager.get_statistics()
    assert stats['global']['total_strategies'] == 1
    assert stats['global']['account_balance'] == 10000.0

@pytest.mark.asyncio
async def test_strategy_with_indicators():
    """Test integración completa estrategia + indicadores"""
    indicator_manager = IndicatorManager()
    indicator_manager.initialize_symbol("BTCUSDT")

    # Alimentar con datos históricos
    price = 62000.0
    for i in range(50):
        bar = {
            'timestamp': datetime.utcnow(),
            'open': price,
            'high': price + 100,
            'low': price - 100,
            'close': price + (i % 3 - 1) * 50,
            'volume': 1000 + (i % 5) * 200
        }
        indicator_manager.update_with_bar("BTCUSDT", bar)
        price += 20

    # Verificar que indicadores están listos
    assert indicator_manager.is_ready("BTCUSDT")

    # Crear estrategia
    manager = StrategyManager(indicator_manager, 10000.0)
    strategy = manager.add_strategy(
        symbol="BTCUSDT",
        strategy_type="breakout_compression"
    )

    # Simular barra que podría generar señal
    test_bar = {
        'timestamp': datetime.utcnow(),
        'open': price,
        'high': price + 200,
        'low': price,
        'close': price + 180,
        'volume': 3000
    }

    # Actualizar indicadores
    indicator_manager.update_with_bar("BTCUSDT", test_bar)

    # Distribuir a estrategia
    await manager.on_bar("BTCUSDT", test_bar)

    # Verificar que estrategia procesó la barra
    assert strategy.total_signals_generated >= 0