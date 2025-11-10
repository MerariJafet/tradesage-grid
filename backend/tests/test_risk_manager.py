import pytest
from datetime import datetime, timedelta
from app.core.risk.risk_manager import RiskManager
from app.core.risk.risk_events import RiskEventType, RiskEventSeverity

class TestRiskManager:
    """Tests para RiskManager"""

    def test_initialization(self):
        """Test inicialización básica"""
        rm = RiskManager(initial_balance=10000.0)

        assert rm.current_balance == 10000.0
        assert rm.trading_enabled == True
        assert rm.kill_switch_active == False
        assert rm.consecutive_losses == 0

    def test_balance_update_no_events(self):
        """Test actualización de balance sin eventos de riesgo"""
        rm = RiskManager(initial_balance=10000.0)

        events = rm.update_balance(10100.0)

        assert rm.current_balance == 10100.0
        assert len(events) == 0

    def test_daily_loss_limit_exceeded(self):
        """Test límite de pérdida diaria excedido"""
        rm = RiskManager(initial_balance=10000.0, max_daily_loss_pct=2.0)

        # Simular pérdida del 3%
        events = rm.update_balance(9700.0)

        assert len(events) == 1
        assert events[0].type == RiskEventType.DAILY_LOSS_LIMIT
        assert events[0].severity == RiskEventSeverity.CRITICAL
        assert rm.kill_switch_active == True

    def test_consecutive_losses_cooldown(self):
        """Test cooldown después de pérdidas consecutivas"""
        rm = RiskManager(max_consecutive_losses=2, cooldown_after_loss_minutes=1)

        # Primera pérdida
        events1 = rm.register_trade_result(-100, "EURUSD", "scalping")
        assert rm.consecutive_losses == 1
        assert rm.trading_enabled == True

        # Segunda pérdida - activar cooldown
        events2 = rm.register_trade_result(-100, "EURUSD", "scalping")
        assert rm.consecutive_losses == 2
        assert rm.trading_enabled == False

        # Verificar cooldown
        can_trade, reason = rm.can_trade()
        assert can_trade == False
        assert "Cooldown active" in reason

    def test_kill_switch_manual(self):
        """Test kill-switch manual"""
        rm = RiskManager()

        rm.manual_kill_switch("Test emergency stop")

        assert rm.kill_switch_active == True
        assert rm.trading_enabled == False
        assert rm.kill_switch_reason == "Test emergency stop"

        # Verificar que no se puede tradear
        can_trade, reason = rm.can_trade()
        assert can_trade == False
        assert "Kill-switch active" in reason

    def test_kill_switch_reset(self):
        """Test reset de kill-switch"""
        rm = RiskManager()

        rm.manual_kill_switch("Test")
        assert rm.kill_switch_active == True

        rm.reset_kill_switch()
        assert rm.kill_switch_active == False
        assert rm.trading_enabled == True

    def test_position_limits_integration(self):
        """Test integración con límites de posición"""
        rm = RiskManager(initial_balance=10000.0)

        # Verificar que se puede abrir posición inicialmente
        can_open, reason = rm.can_open_position("EURUSD", 1000, 50)
        assert can_open == True

        # Registrar posición abierta
        rm.register_position_opened("pos1", "EURUSD", 50)

        # Verificar estadísticas
        stats = rm.get_statistics()
        assert stats["position_limits"]["exposure"]["total"] == 50
        assert stats["position_limits"]["positions"]["open"] == 1

    def test_statistics_comprehensive(self):
        """Test estadísticas completas"""
        rm = RiskManager(initial_balance=10000.0)

        # Simular algunas operaciones
        rm.update_balance(9800.0)  # Pérdida de 200
        rm.register_trade_result(-100, "EURUSD", "scalping")
        rm.register_position_opened("pos1", "EURUSD", 50)

        stats = rm.get_statistics()

        # Verificar estructura
        assert "status" in stats
        assert "balance" in stats
        assert "period_pnl" in stats
        assert "drawdown" in stats
        assert "position_limits" in stats
        assert "risk_events" in stats

        # Verificar valores
        assert stats["balance"]["current"] == 9800.0
        assert stats["balance"]["total_pnl"] == -200.0
        assert stats["status"]["consecutive_losses"] == 1