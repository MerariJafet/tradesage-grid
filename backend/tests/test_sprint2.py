import pytest
import asyncio
from app.core.exchanges.binance.ws_client import BinanceWSClient
from app.core.data.normalizer import DataNormalizer
from app.core.data.sequence_validator import SequenceValidator
from app.core.data.writer import DataWriter
from app.core.monitoring.latency_tracker import LatencyTracker

@pytest.mark.asyncio
async def test_binance_ws_connection():
    """Test conexión básica a Binance WebSocket"""
    client = BinanceWSClient(market_type="spot", symbols=["BTCUSDT"])

    received_tick = False

    async def tick_callback(tick):
        nonlocal received_tick
        received_tick = True
        assert 'symbol' in tick
        assert 'price' in tick

    client.subscribe("tick", tick_callback)

    # Conectar y esperar 5 segundos
    task = asyncio.create_task(client.connect())
    await asyncio.sleep(5)

    assert client.is_connected
    assert received_tick

    await client.close()

@pytest.mark.asyncio
async def test_auto_reconnect():
    """Test reconexión automática"""
    client = BinanceWSClient(market_type="spot", symbols=["BTCUSDT"])

    # Conectar
    task = asyncio.create_task(client.connect())
    await asyncio.sleep(2)

    # Forzar desconexión
    if client.websocket:
        await client.websocket.close()

    # Esperar reconexión
    await asyncio.sleep(5)

    # Debe haberse reconectado
    assert client.is_connected
    assert client.reconnect_attempts >= 1

    await client.close()

def test_data_normalizer_validation():
    """Test validación de datos"""
    normalizer = DataNormalizer()

    # Tick válido
    valid_tick = {
        "timestamp": "2025-10-15T10:30:00Z",
        "symbol": "BTCUSDT",
        "price": 62000,
        "quantity": 0.5
    }
    assert normalizer.validate_tick(valid_tick) == True

    # Tick inválido (falta price)
    invalid_tick = {
        "timestamp": "2025-10-15T10:30:00Z",
        "symbol": "BTCUSDT",
        "quantity": 0.5
    }
    assert normalizer.validate_tick(invalid_tick) == False

def test_sequence_validator():
    """Test detección de gaps en secuencias"""
    validator = SequenceValidator()

    # Secuencia normal
    assert validator.validate("BTCUSDT", 100) == True
    assert validator.validate("BTCUSDT", 101) == True
    assert validator.validate("BTCUSDT", 102) == True

    # Gap detectado
    assert validator.validate("BTCUSDT", 105) == False
    assert validator.gaps_detected == 1

def test_latency_tracker():
    """Test tracking de latencia"""
    tracker = LatencyTracker(window_size=100)

    # Registrar latencias
    for i in range(100):
        tracker.record("binance", "BTCUSDT", 50 + i * 0.5)

    stats = tracker.get_stats("binance", "BTCUSDT")

    assert stats['count'] == 100
    assert 70 < stats['p50'] < 80
    assert 90 < stats['p95'] < 100
    assert stats['min'] == 50

    # Test degradación
    for i in range(50):
        tracker.record("binance", "ETHUSDT", 150)

    assert tracker.check_degradation("binance", "ETHUSDT", threshold_p95=100) == True

@pytest.mark.asyncio
async def test_data_writer_batch():
    """Test batch insert del writer"""
    writer = DataWriter()

    # Añadir ticks
    for i in range(150):
        tick = {
            "timestamp": "2025-10-15T10:30:00Z",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "price": 62000 + i,
            "quantity": 0.5,
            "is_buyer_maker": True,
            "trade_id": 1000 + i
        }
        await writer.add_tick(tick)

    # Debe haber flusheado al menos una vez (batch_size=100)
    assert writer.ticks_written >= 100

    # Flush final
    await writer.flush_all()
    assert writer.ticks_written == 150

@pytest.mark.asyncio
async def test_ws_manager_integration():
    """Test integración completa del manager"""
    from app.core.ws_manager import WebSocketManager

    manager = WebSocketManager(symbols=["BTCUSDT"])

    # Iniciar en background
    task = asyncio.create_task(manager.start())

    # Esperar 10 segundos
    await asyncio.sleep(10)

    # Verificar estado
    status = await manager.get_status()
    assert status['is_running'] == True
    assert len(status['clients']) == 2  # spot + futures
    assert status['writer_stats']['ticks_written'] > 0

    # Detener
    await manager.stop()
    assert status['is_running'] == False