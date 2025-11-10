from collections import deque
from typing import Dict, List
import time
from app.utils.logger import get_logger
from app.api.routes.metrics import ws_latency

logger = get_logger("latency_tracker")

class LatencyTracker:
    """Track latency statistics por exchange y símbolo"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        # Deque por (exchange, symbol)
        self.latencies: Dict[str, deque] = {}

    def record(self, exchange: str, symbol: str, latency_ms: float):
        """Registrar latencia"""
        key = f"{exchange}:{symbol}"

        if key not in self.latencies:
            self.latencies[key] = deque(maxlen=self.window_size)

        self.latencies[key].append(latency_ms)

        # Actualizar métrica Prometheus
        ws_latency.labels(exchange=exchange, symbol=symbol).observe(latency_ms / 1000)

    def get_stats(self, exchange: str, symbol: str) -> Dict:
        """Obtener estadísticas de latencia"""
        key = f"{exchange}:{symbol}"

        if key not in self.latencies or not self.latencies[key]:
            return {
                "count": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "avg": 0,
                "max": 0
            }

        latencies = sorted(self.latencies[key])
        count = len(latencies)

        return {
            "count": count,
            "p50": latencies[int(count * 0.5)],
            "p95": latencies[int(count * 0.95)],
            "p99": latencies[int(count * 0.99)],
            "avg": sum(latencies) / count,
            "max": latencies[-1],
            "min": latencies[0]
        }

    def get_all_stats(self) -> Dict:
        """Obtener estadísticas de todos los símbolos"""
        all_stats = {}
        for key in self.latencies.keys():
            exchange, symbol = key.split(":")
            all_stats[key] = self.get_stats(exchange, symbol)
        return all_stats

    def check_degradation(self, exchange: str, symbol: str, threshold_p95: float = 100) -> bool:
        """
        Verificar si latencia está degradada
        Returns: True si p95 > threshold_p95 ms
        """
        stats = self.get_stats(exchange, symbol)
        if stats['count'] < 10:  # Necesitamos suficientes muestras
            return False

        if stats['p95'] > threshold_p95:
            logger.warning(
                "latency_degradation_detected",
                exchange=exchange,
                symbol=symbol,
                p95=stats['p95'],
                threshold=threshold_p95
            )
            return True

        return False

# Instancia global del tracker
latency_tracker = LatencyTracker()