import redis
import pickle
from typing import Optional, Any
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("indicator_cache")

class IndicatorCache:
    """
    Caché en Redis para indicadores técnicos
    - TTL configurable por indicador
    - Serialización con pickle
    - Keys: indicator:{symbol}:{indicator_name}:{params}
    """

    def __init__(self):
        self.redis_client = None
        self.cache_enabled = False

        try:
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=False  # Para pickle
            )
            client.ping()
            self.redis_client = client
            self.cache_enabled = True
            logger.info(
                "indicator_cache_connected",
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT
            )
            print("INFO: Conexión a Redis exitosa. Caché activa.")
        except redis.exceptions.ConnectionError:
            self.redis_client = None
            self.cache_enabled = False
            logger.warning(
                "indicator_cache_connection_failed",
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT
            )
            print("WARNING: Falló la conexión a Redis. Caché deshabilitada. El backtest continuará sin caché.")

    def _make_key(self, symbol: str, indicator_name: str, params: str = "") -> str:
        """Generar clave para Redis"""
        key = f"indicator:{symbol}:{indicator_name}"
        if params:
            key += f":{params}"
        return key

    def set(
        self,
        symbol: str,
        indicator_name: str,
        value: Any,
        ttl: int = 60
    ) -> bool:
        """Guardar valor en caché con TTL en segundos"""
        key = self._make_key(symbol, indicator_name)

        try:
            if not self.cache_enabled or not self.redis_client:
                return False
            serialized = pickle.dumps(value)
            success = self.redis_client.setex(key, ttl, serialized)

            if success:
                logger.debug(
                    "indicator_cached",
                    symbol=symbol,
                    indicator=indicator_name,
                    ttl=ttl
                )
            else:
                logger.warning(
                    "cache_set_failed",
                    symbol=symbol,
                    indicator=indicator_name
                )

            return bool(success)

        except Exception as e:
            logger.error(
                "cache_set_error",
                symbol=symbol,
                indicator=indicator_name,
                error=str(e)
            )
            return False

    def get(
        self,
        symbol: str,
        indicator_name: str,
        params: str = ""
    ) -> Optional[Any]:
        """Obtener valor desde caché"""
        key = self._make_key(symbol, indicator_name, params)

        try:
            if not self.cache_enabled or not self.redis_client:
                return None
            serialized = self.redis_client.get(key)
            if serialized:
                value = pickle.loads(serialized)
                logger.debug(
                    "indicator_cache_hit",
                    symbol=symbol,
                    indicator=indicator_name
                )
                return value

            logger.debug(
                "indicator_cache_miss",
                symbol=symbol,
                indicator=indicator_name
            )
            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e), key=key)
            return None

    def delete(self, symbol: str, indicator_name: str, params: str = ""):
        """Eliminar indicador del caché"""
        key = self._make_key(symbol, indicator_name, params)
        try:
            if not self.cache_enabled or not self.redis_client:
                return
            self.redis_client.delete(key)
            logger.debug("indicator_cache_deleted", key=key)
        except Exception as e:
            logger.error("cache_delete_error", error=str(e), key=key)

    def clear_symbol(self, symbol: str):
        """Limpiar todos los indicadores de un símbolo"""
        pattern = f"indicator:{symbol}:*"
        try:
            if not self.cache_enabled or not self.redis_client:
                return
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info("symbol_indicators_cleared", symbol=symbol, count=len(keys))
        except Exception as e:
            logger.error("cache_clear_error", error=str(e), symbol=symbol)

    @property
    def redis(self):
        """Redis client instance"""
        return self.redis_client