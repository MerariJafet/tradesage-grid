# backend/app/core/telemetry/performance_tracker.py

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import asyncio
from collections import defaultdict
from app.core.telemetry.sensors import (
    LatencySensor,
    ExecutionQualitySensor,
    MarketDataSensor,
    SystemResourceSensor
)
from app.utils.logger import get_logger

logger = get_logger("performance_tracker")

class PerformanceTracker:
    """
    Tracker central de rendimiento del sistema de trading

    Mide y monitorea:
    - Latencias de operaciones críticas
    - Calidad de ejecución de órdenes
    - Calidad de datos de mercado
    - Recursos del sistema
    - Rendimiento de estrategias
    """

    def __init__(self):
        # Sensores de latencia
        self.latency_sensors: Dict[str, LatencySensor] = {
            "order_to_fill": LatencySensor("order_to_fill"),
            "signal_to_order": LatencySensor("signal_to_order"),
            "bar_processing": LatencySensor("bar_processing"),
            "indicator_calculation": LatencySensor("indicator_calculation")
        }

        # Sensores de calidad de ejecución
        self.execution_sensors: Dict[str, ExecutionQualitySensor] = {}

        # Sensores de datos de mercado
        self.market_data_sensors: Dict[str, MarketDataSensor] = {}

        # Sensor de recursos del sistema
        self.system_sensor = SystemResourceSensor()

        # Estadísticas de estrategias
        self.strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Estado del tracker
        self.is_tracking = False
        self.start_time = datetime.now(timezone.utc)

    def get_latency_sensor(self, name: str) -> LatencySensor:
        """Obtener sensor de latencia"""
        if name not in self.latency_sensors:
            self.latency_sensors[name] = LatencySensor(name)
        return self.latency_sensors[name]

    def get_execution_sensor(self, symbol: str) -> ExecutionQualitySensor:
        """Obtener sensor de calidad de ejecución"""
        if symbol not in self.execution_sensors:
            self.execution_sensors[symbol] = ExecutionQualitySensor(symbol)
        return self.execution_sensors[symbol]

    def get_market_data_sensor(self, symbol: str) -> MarketDataSensor:
        """Obtener sensor de datos de mercado"""
        if symbol not in self.market_data_sensors:
            self.market_data_sensors[symbol] = MarketDataSensor(symbol)
        return self.market_data_sensors[symbol]

    def record_strategy_performance(
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
        """Registrar rendimiento de estrategia"""

        self.strategy_stats[strategy_name] = {
            "symbol": symbol,
            "signals_generated": signals_generated,
            "signals_executed": signals_executed,
            "trades": trades,
            "winning_trades": winning_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "last_update": datetime.now(timezone.utc)
        }

    def record_bar_processing(self, symbol: str, processing_time_ms: float):
        """Registrar tiempo de procesamiento de barra"""
        sensor = self.get_latency_sensor("bar_processing")
        # Simular start/end para registro directo
        sensor.samples.append(processing_time_ms)
        if len(sensor.samples) > sensor.max_samples:
            sensor.samples.pop(0)

    def record_indicator_calculation(self, indicator_name: str, calculation_time_ms: float):
        """Registrar tiempo de cálculo de indicador"""
        sensor = self.get_latency_sensor(f"indicator_{indicator_name}")
        sensor.samples.append(calculation_time_ms)
        if len(sensor.samples) > sensor.max_samples:
            sensor.samples.pop(0)

    async def start_system_monitoring(self):
        """Iniciar monitoreo de recursos del sistema"""
        self.is_tracking = True

        async def monitor_loop():
            while self.is_tracking:
                await self.system_sensor.record_system_stats()
                await asyncio.sleep(60)  # Cada minuto

        asyncio.create_task(monitor_loop())
        logger.info("system_monitoring_started")

    def stop_system_monitoring(self):
        """Detener monitoreo de recursos del sistema"""
        self.is_tracking = False
        logger.info("system_monitoring_stopped")

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de rendimiento"""

        # Estadísticas de latencia
        latency_stats = {}
        for name, sensor in self.latency_sensors.items():
            latency_stats[name] = sensor.get_stats()

        # Estadísticas de ejecución
        execution_stats = {}
        for symbol, sensor in self.execution_sensors.items():
            execution_stats[symbol] = sensor.get_stats()

        # Estadísticas de datos de mercado
        market_data_stats = {}
        for symbol, sensor in self.market_data_sensors.items():
            market_data_stats[symbol] = sensor.get_stats()

        # Estadísticas de estrategias
        strategy_performance = dict(self.strategy_stats)

        # Estadísticas del sistema
        system_stats = self.system_sensor.get_stats()

        # Métricas globales
        total_signals = sum(s.get("signals_generated", 0) for s in strategy_performance.values())
        total_trades = sum(s.get("trades", 0) for s in strategy_performance.values())
        total_pnl = sum(s.get("total_pnl", 0) for s in strategy_performance.values())

        return {
            "timestamp": datetime.now(timezone.utc),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),

            "latency": latency_stats,
            "execution_quality": execution_stats,
            "market_data": market_data_stats,
            "system_resources": system_stats,
            "strategy_performance": strategy_performance,

            "global_metrics": {
                "total_strategies": len(strategy_performance),
                "total_signals_generated": total_signals,
                "total_trades_executed": total_trades,
                "total_pnl": total_pnl,
                "avg_win_rate": sum(s.get("win_rate", 0) for s in strategy_performance.values()) / len(strategy_performance) if strategy_performance else 0
            }
        }

    def get_health_score(self) -> Dict[str, Any]:
        """
        Calcular score de salud del sistema (0-100)

        Factores:
        - Latencia de órdenes (< 100ms = 100 puntos)
        - Tasa de rechazo de órdenes (< 5% = 100 puntos)
        - Calidad de datos (sin gaps = 100 puntos)
        - Recursos del sistema (< 80% uso = 100 puntos)
        """

        score = 100
        issues = []

        # Evaluar latencia de órdenes
        order_latency = self.latency_sensors.get("order_to_fill")
        if order_latency and order_latency.samples:
            avg_latency = order_latency.get_stats()["avg"]
            if avg_latency > 100:  # > 100ms
                penalty = min(50, (avg_latency - 100) / 10)  # Máximo 50 puntos de penalización
                score -= penalty
                issues.append(f"High order latency: {avg_latency:.1f}ms")

        # Evaluar tasa de rechazo
        total_rejected = 0
        total_orders = 0
        for sensor in self.execution_sensors.values():
            stats = sensor.get_stats(hours=1)  # Última hora
            total_rejected += stats["rejected_orders"]
            total_orders += stats["total_orders"]

        if total_orders > 0:
            rejection_rate = total_rejected / total_orders * 100
            if rejection_rate > 5:  # > 5%
                penalty = min(30, rejection_rate - 5)  # Máximo 30 puntos de penalización
                score -= penalty
                issues.append(f"High rejection rate: {rejection_rate:.1f}%")

        # Evaluar calidad de datos
        total_gaps = sum(sensor.sequence_gaps for sensor in self.market_data_sensors.values())
        if total_gaps > 10:  # Más de 10 gaps
            penalty = min(20, total_gaps - 10)  # Máximo 20 puntos de penalización
            score -= penalty
            issues.append(f"Data quality issues: {total_gaps} sequence gaps")

        # Evaluar recursos del sistema
        system_stats = self.system_sensor.get_stats()
        if system_stats["avg_cpu_percent"] > 80 or system_stats["avg_memory_percent"] > 80:
            score -= 10
            issues.append("High system resource usage")

        return {
            "score": max(0, score),  # No negativo
            "status": "healthy" if score >= 80 else "warning" if score >= 60 else "critical",
            "issues": issues
        }

    def reset_stats(self):
        """Resetear todas las estadísticas"""
        for sensor in self.latency_sensors.values():
            sensor.samples.clear()
            sensor.start_times.clear()

        for sensor in self.execution_sensors.values():
            sensor.executions.clear()

        for sensor in self.market_data_sensors.values():
            sensor.bar_delays.clear()
            sensor.tick_delays.clear()
            sensor.sequence_gaps = 0

        self.strategy_stats.clear()
        self.start_time = datetime.now(timezone.utc)

        logger.info("performance_stats_reset")