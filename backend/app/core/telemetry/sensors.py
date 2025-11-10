# backend/app/core/telemetry/sensors.py

from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import statistics
import asyncio
from app.utils.logger import get_logger

logger = get_logger("telemetry_sensors")

class LatencySensor:
    """
    Sensor para medir latencias de operaciones críticas
    """

    def __init__(self, name: str, max_samples: int = 1000):
        self.name = name
        self.max_samples = max_samples
        self.samples: List[float] = []
        self.start_times: Dict[str, datetime] = {}

    def start(self, operation_id: str):
        """Iniciar medición de latencia"""
        self.start_times[operation_id] = datetime.utcnow()

    def end(self, operation_id: str) -> Optional[float]:
        """Finalizar medición y retornar latencia en ms"""
        if operation_id not in self.start_times:
            logger.warning(f"operation_not_started", operation_id=operation_id, sensor=self.name)
            return None

        start_time = self.start_times.pop(operation_id)
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms

        # Almacenar muestra
        self.samples.append(latency)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

        return latency

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de latencia"""
        if not self.samples:
            return {
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            }

        sorted_samples = sorted(self.samples)

        return {
            "count": len(self.samples),
            "avg": statistics.mean(self.samples),
            "min": min(self.samples),
            "max": max(self.samples),
            "p50": statistics.median(sorted_samples),
            "p95": sorted_samples[int(len(sorted_samples) * 0.95)],
            "p99": sorted_samples[int(len(sorted_samples) * 0.99)]
        }

class ExecutionQualitySensor:
    """
    Sensor para medir calidad de ejecución de órdenes
    """

    def __init__(self, symbol: str, max_samples: int = 1000):
        self.symbol = symbol
        self.max_samples = max_samples
        self.executions: List[Dict[str, Any]] = []

    def record_order(
        self,
        expected_price: float,
        filled_price: float,
        expected_quantity: float,
        filled_quantity: float,
        rejected: bool = False,
        rejection_reason: Optional[str] = None
    ):
        """Registrar ejecución de orden"""

        execution = {
            "timestamp": datetime.utcnow(),
            "expected_price": expected_price,
            "filled_price": filled_price,
            "expected_quantity": expected_quantity,
            "filled_quantity": filled_quantity,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "price_slippage": (filled_price - expected_price) / expected_price * 100 if not rejected else 0,
            "quantity_fill_rate": filled_quantity / expected_quantity * 100 if expected_quantity > 0 else 0
        }

        self.executions.append(execution)
        if len(self.executions) > self.max_samples:
            self.executions.pop(0)

        logger.debug(
            "execution_recorded",
            symbol=self.symbol,
            rejected=rejected,
            price_slippage=execution["price_slippage"],
            fill_rate=execution["quantity_fill_rate"]
        )

    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Obtener estadísticas de calidad de ejecución"""

        # Filtrar por tiempo
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_executions = [e for e in self.executions if e["timestamp"] > cutoff]

        if not recent_executions:
            return {
                "total_orders": 0,
                "rejected_orders": 0,
                "rejection_rate": 0,
                "avg_price_slippage": 0,
                "avg_fill_rate": 0,
                "slippage_std": 0,
                "fill_rate_std": 0
            }

        # Calcular métricas
        successful_executions = [e for e in recent_executions if not e["rejected"]]
        rejected_count = len(recent_executions) - len(successful_executions)

        price_slippages = [e["price_slippage"] for e in successful_executions]
        fill_rates = [e["quantity_fill_rate"] for e in successful_executions]

        return {
            "total_orders": len(recent_executions),
            "rejected_orders": rejected_count,
            "rejection_rate": rejected_count / len(recent_executions) * 100,
            "avg_price_slippage": statistics.mean(price_slippages) if price_slippages else 0,
            "avg_fill_rate": statistics.mean(fill_rates) if fill_rates else 0,
            "slippage_std": statistics.stdev(price_slippages) if len(price_slippages) > 1 else 0,
            "fill_rate_std": statistics.stdev(fill_rates) if len(fill_rates) > 1 else 0
        }

class MarketDataSensor:
    """
    Sensor para medir calidad y latencia de datos de mercado
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.last_bar_time: Optional[datetime] = None
        self.last_tick_time: Optional[datetime] = None
        self.bar_delays: List[float] = []
        self.tick_delays: List[float] = []
        self.sequence_gaps = 0
        self.max_samples = 1000

    def record_bar(self, bar_timestamp: datetime):
        """Registrar recepción de barra"""
        now = datetime.utcnow()
        delay = (now - bar_timestamp).total_seconds() * 1000  # ms

        self.bar_delays.append(delay)
        if len(self.bar_delays) > self.max_samples:
            self.bar_delays.pop(0)

        self.last_bar_time = now

    def record_tick(self, tick_timestamp: datetime):
        """Registrar recepción de tick"""
        now = datetime.utcnow()
        delay = (now - tick_timestamp).total_seconds() * 1000  # ms

        self.tick_delays.append(delay)
        if len(self.tick_delays) > self.max_samples:
            self.tick_delays.pop(0)

        self.last_tick_time = now

    def record_sequence_gap(self):
        """Registrar gap en secuencia"""
        self.sequence_gaps += 1

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de calidad de datos"""

        return {
            "symbol": self.symbol,
            "last_bar_age_seconds": (datetime.utcnow() - self.last_bar_time).total_seconds() if self.last_bar_time else None,
            "last_tick_age_seconds": (datetime.utcnow() - self.last_tick_time).total_seconds() if self.last_tick_time else None,
            "avg_bar_delay_ms": statistics.mean(self.bar_delays) if self.bar_delays else 0,
            "avg_tick_delay_ms": statistics.mean(self.tick_delays) if self.tick_delays else 0,
            "sequence_gaps": self.sequence_gaps,
            "bar_delays_count": len(self.bar_delays),
            "tick_delays_count": len(self.tick_delays)
        }

class SystemResourceSensor:
    """
    Sensor para medir recursos del sistema
    """

    def __init__(self):
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.max_samples = 100

    async def record_system_stats(self):
        """Registrar estadísticas del sistema"""
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent

            self.cpu_usage.append(cpu)
            self.memory_usage.append(memory)

            if len(self.cpu_usage) > self.max_samples:
                self.cpu_usage.pop(0)
            if len(self.memory_usage) > self.max_samples:
                self.memory_usage.pop(0)

        except ImportError:
            # psutil no disponible
            pass
        except Exception as e:
            logger.error("system_stats_error", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de recursos"""

        return {
            "avg_cpu_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_percent": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "max_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0,
            "max_memory_percent": max(self.memory_usage) if self.memory_usage else 0,
            "samples_count": len(self.cpu_usage)
        }