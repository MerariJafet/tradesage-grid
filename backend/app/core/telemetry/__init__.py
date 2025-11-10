# backend/app/core/telemetry/__init__.py

"""
Telemetry System Module

Sistema de telemetría y monitoreo para medir:
- Latencias de operaciones críticas
- Calidad de ejecución de órdenes
- Rendimiento de estrategias
- Salud del sistema
- Recursos del sistema
"""

from .sensors import LatencySensor, ExecutionQualitySensor, MarketDataSensor, SystemResourceSensor
from .performance_tracker import PerformanceTracker
from .metrics import TelemetrySystem, telemetry_system, init_telemetry_system

__all__ = [
    "LatencySensor",
    "ExecutionQualitySensor",
    "MarketDataSensor",
    "SystemResourceSensor",
    "PerformanceTracker",
    "TelemetrySystem",
    "telemetry_system",
    "init_telemetry_system"
]