# backend/app/core/telemetry/metrics.py

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from app.core.telemetry.performance_tracker import PerformanceTracker
from app.utils.logger import get_logger

logger = get_logger("telemetry_metrics")

# Instancia global del sistema de telemetría
telemetry_system = None

def init_telemetry_system() -> 'TelemetrySystem':
    """Inicializar sistema de telemetría"""
    global telemetry_system
    if telemetry_system is None:
        telemetry_system = TelemetrySystem()
    return telemetry_system

class TelemetrySystem:
    """
    Sistema central de telemetría y métricas

    Proporciona interfaz unificada para:
    - Medición de latencias
    - Monitoreo de calidad de ejecución
    - Seguimiento de rendimiento
    - Reportes de salud del sistema
    """

    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.is_enabled = True
        self.start_time = datetime.utcnow()

        logger.info("telemetry_system_initialized")

    def get_latency_sensor(self, name: str):
        """Obtener sensor de latencia"""
        return self.performance_tracker.get_latency_sensor(name)

    def get_execution_sensor(self, symbol: str):
        """Obtener sensor de calidad de ejecución"""
        return self.performance_tracker.get_execution_sensor(symbol)

    def get_market_data_sensor(self, symbol: str):
        """Obtener sensor de datos de mercado"""
        return self.performance_tracker.get_market_data_sensor(symbol)

    def record_signal_generated(self, strategy_name: str, symbol: str):
        """Registrar generación de señal"""
        if not self.is_enabled:
            return

        # Podríamos añadir métricas específicas de señales aquí
        logger.debug("signal_generated", strategy=strategy_name, symbol=symbol)

    def record_order_submitted(
        self,
        order_id: str,
        strategy_name: str,
        symbol: str,
        side: str,
        quantity: float
    ):
        """Registrar envío de orden"""
        if not self.is_enabled:
            return

        # Iniciar medición de latencia signal_to_order
        latency_sensor = self.get_latency_sensor("signal_to_order")
        latency_sensor.start(f"order_{order_id}")

        logger.debug(
            "order_submitted",
            order_id=order_id,
            strategy=strategy_name,
            symbol=symbol,
            side=side,
            quantity=quantity
        )

    def record_order_filled(
        self,
        order_id: str,
        filled_price: float,
        filled_quantity: float,
        slippage: float,
        commission: float
    ):
        """Registrar ejecución de orden"""
        if not self.is_enabled:
            return

        # Finalizar medición de latencia
        latency_sensor = self.get_latency_sensor("signal_to_order")
        latency = latency_sensor.end(f"order_{order_id}")

        if latency:
            logger.debug(
                "order_latency_measured",
                order_id=order_id,
                latency_ms=latency
            )

        # Registrar en sensor de calidad de ejecución
        # Nota: Esto se hace desde paper_exchange.py

    def record_bar_processed(self, symbol: str, processing_time_ms: float):
        """Registrar procesamiento de barra"""
        if not self.is_enabled:
            return

        self.performance_tracker.record_bar_processing(symbol, processing_time_ms)

    def record_indicator_calculated(self, indicator_name: str, calculation_time_ms: float):
        """Registrar cálculo de indicador"""
        if not self.is_enabled:
            return

        self.performance_tracker.record_indicator_calculation(indicator_name, calculation_time_ms)

    def record_strategy_stats(
        self,
        strategy_name: str,
        symbol: str,
        signals_generated: int,
        signals_executed: int,
        trades: int,
        winning_trades: int,
        total_pnl: float,
        win_rate: float
    ):
        """Registrar estadísticas de estrategia"""
        if not self.is_enabled:
            return

        self.performance_tracker.record_strategy_performance(
            strategy_name=strategy_name,
            symbol=symbol,
            signals_generated=signals_generated,
            signals_executed=signals_executed,
            trades=trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            win_rate=win_rate
        )

    async def start_monitoring(self):
        """Iniciar monitoreo del sistema"""
        await self.performance_tracker.start_system_monitoring()

    def stop_monitoring(self):
        """Detener monitoreo del sistema"""
        self.performance_tracker.stop_system_monitoring()

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de telemetría"""
        return self.performance_tracker.get_comprehensive_report()

    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud del sistema"""
        health = self.performance_tracker.get_health_score()

        return {
            "timestamp": datetime.utcnow(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "health_score": health["score"],
            "status": health["status"],
            "issues": health["issues"],
            "is_enabled": self.is_enabled
        }

    def get_latency_report(self) -> Dict[str, Any]:
        """Obtener reporte de latencias"""
        latency_stats = {}
        for name, sensor in self.performance_tracker.latency_sensors.items():
            latency_stats[name] = sensor.get_stats()

        return {
            "timestamp": datetime.utcnow(),
            "latencies": latency_stats
        }

    def get_execution_report(self) -> Dict[str, Any]:
        """Obtener reporte de calidad de ejecución"""
        execution_stats = {}
        for symbol, sensor in self.performance_tracker.execution_sensors.items():
            execution_stats[symbol] = sensor.get_stats()

        return {
            "timestamp": datetime.utcnow(),
            "execution_quality": execution_stats
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de rendimiento de estrategias"""
        return {
            "timestamp": datetime.utcnow(),
            "strategy_performance": dict(self.performance_tracker.strategy_stats)
        }

    def enable(self):
        """Habilitar telemetría"""
        self.is_enabled = True
        logger.info("telemetry_enabled")

    def disable(self):
        """Deshabilitar telemetría"""
        self.is_enabled = False
        logger.info("telemetry_disabled")

    def reset(self):
        """Resetear todas las métricas"""
        self.performance_tracker.reset_stats()
        self.start_time = datetime.utcnow()
        logger.info("telemetry_reset")

# Inicializar sistema global
telemetry_system = TelemetrySystem()