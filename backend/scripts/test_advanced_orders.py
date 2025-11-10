# backend/scripts/test_advanced_orders.py

import asyncio
from datetime import datetime
from app.core.orders import (
    LimitOrder, StopLimitOrder, OCOOrder, IcebergOrder, TrailingStopOrder, OrderManager
)
from app.core.orders.base_order import OrderSide

async def test_limit_order():
    """Test limit order functionality"""
    print("Testing Limit Order...")

    # Create buy limit order
    buy_order = LimitOrder(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        limit_price=50000.0
    )

    # Test triggering
    assert not buy_order.should_trigger(51000.0, {})  # Price above limit
    assert buy_order.should_trigger(49000.0, {})      # Price at limit
    assert buy_order.should_trigger(48000.0, {})      # Price below limit

    # Create sell limit order
    sell_order = LimitOrder(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        quantity=1.0,
        limit_price=52000.0
    )

    assert not sell_order.should_trigger(51000.0, {})  # Price below limit
    assert sell_order.should_trigger(53000.0, {})      # Price at limit
    assert sell_order.should_trigger(54000.0, {})      # Price above limit

    print("✅ Limit Order tests passed")

async def test_stop_limit_order():
    """Test stop-limit order functionality"""
    print("Testing Stop-Limit Order...")

    # Buy stop-limit
    buy_stop_limit = StopLimitOrder(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        stop_price=51000.0,
        limit_price=51500.0
    )

    # Before stop trigger
    assert not buy_stop_limit.should_trigger(50500.0, {})

    # After stop trigger, should behave like limit
    assert buy_stop_limit.should_trigger(51200.0, {})  # Above stop, below limit
    assert not buy_stop_limit.should_trigger(51600.0, {})  # Above limit

    print("✅ Stop-Limit Order tests passed")

async def test_trailing_stop_order():
    """Test trailing stop order functionality"""
    print("Testing Trailing Stop Order...")

    # Sell trailing stop
    trailing_stop = TrailingStopOrder(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        quantity=1.0,
        trailing_percent=2.0  # 2% trailing
    )

    # Set initial price
    trailing_stop.set_initial_price(50000.0)
    assert trailing_stop.stop_price == 50000.0 * 0.98  # 49000

    # Price moves up, stop should trail
    trailing_stop.should_trigger(51000.0, {})  # Updates highest price
    assert trailing_stop.stop_price == 51000.0 * 0.98  # 49980

    # Price drops to stop level
    assert trailing_stop.should_trigger(49900.0, {})  # Should trigger

    print("✅ Trailing Stop Order tests passed")

async def test_order_manager():
    """Test order manager functionality"""
    print("Testing Order Manager...")

    manager = OrderManager()

    # Add orders
    limit_order = LimitOrder("BTCUSDT", OrderSide.BUY, 1.0, 50000.0)
    order_id = manager.add_order(limit_order)

    assert len(manager.get_active_orders()) == 1
    assert len(manager.get_active_orders("BTCUSDT")) == 1

    # Process market data
    triggered = manager.process_market_data("BTCUSDT", 49000.0, {})
    assert len(triggered) == 1
    assert len(manager.get_active_orders()) == 0
    assert len(manager.filled_orders) == 1

    print("✅ Order Manager tests passed")

async def main():
    """Run all tests"""
    print("🧪 Testing Advanced Orders Module\n")

    try:
        await test_limit_order()
        await test_stop_limit_order()
        await test_trailing_stop_order()
        await test_order_manager()

        print("\n🎉 All Advanced Orders tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())