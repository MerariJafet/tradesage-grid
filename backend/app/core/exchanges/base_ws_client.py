from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
import asyncio
import websockets
import json
from app.utils.logger import get_logger

logger = get_logger("ws_client")

class BaseWSClient(ABC):
    def __init__(self, url: str, streams: List[str]):
        self.url = url
        self.streams = streams
        self.websocket = None
        self.is_connected = False
        self.callbacks: Dict[str, List[Callable]] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1  # segundos, exponential backoff
        self.ping_interval = 180  # 3 minutos
        self.last_ping = None

    @abstractmethod
    async def parse_message(self, message: dict):
        """Parse mensaje específico del exchange"""
        pass

    def subscribe(self, event_type: str, callback: Callable):
        """Registrar callback para tipo de evento"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        logger.info("callback_registered", event_type=event_type)

    async def emit(self, event_type: str, data: dict):
        """Emitir evento a todos los callbacks registrados"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(
                        "callback_error",
                        event_type=event_type,
                        error=str(e)
                    )

    async def connect(self):
        """Conectar al WebSocket con auto-reconnect"""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Construir URL con streams
                stream_names = "/".join(self.streams)
                full_url = f"{self.url}/{stream_names}"

                logger.info(
                    "connecting_to_websocket",
                    url=self.url,
                    streams=self.streams,
                    attempt=self.reconnect_attempts + 1
                )

                self.websocket = await websockets.connect(full_url)
                self.is_connected = True
                self.reconnect_attempts = 0
                self.reconnect_delay = 1

                logger.info("websocket_connected", streams=self.streams)

                # Iniciar tareas en paralelo
                await asyncio.gather(
                    self._listen(),
                    self._ping_loop()
                )

            except websockets.exceptions.ConnectionClosed:
                logger.warning("connection_closed", streams=self.streams)
                await self._handle_reconnect()

            except Exception as e:
                logger.error(
                    "connection_error",
                    error=str(e),
                    streams=self.streams
                )
                await self._handle_reconnect()

    async def _listen(self):
        """Escuchar mensajes del WebSocket"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.parse_message(data)
                except json.JSONDecodeError as e:
                    logger.error("json_decode_error", error=str(e))
                except Exception as e:
                    logger.error("message_processing_error", error=str(e))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("listen_connection_closed")
            self.is_connected = False

    async def _ping_loop(self):
        """Enviar pings periódicos para mantener conexión"""
        while self.is_connected:
            try:
                await asyncio.sleep(self.ping_interval)
                if self.websocket and self.is_connected:
                    await self.websocket.ping()
                    self.last_ping = asyncio.get_event_loop().time()
                    logger.debug("ping_sent", streams=self.streams)
            except Exception as e:
                logger.error("ping_error", error=str(e))

    async def _handle_reconnect(self):
        """Manejar reconexión con exponential backoff"""
        self.is_connected = False
        self.reconnect_attempts += 1

        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                "max_reconnect_attempts_reached",
                attempts=self.reconnect_attempts
            )
            await self.emit("max_reconnects_reached", {})
            return

        wait_time = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        wait_time = min(wait_time, 60)  # Max 60 segundos

        logger.info(
            "reconnecting",
            attempt=self.reconnect_attempts,
            wait_seconds=wait_time
        )

        await asyncio.sleep(wait_time)

    async def close(self):
        """Cerrar conexión limpiamente"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
            logger.info("websocket_closed", streams=self.streams)