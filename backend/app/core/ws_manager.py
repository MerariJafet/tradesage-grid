from app.core.exchanges.binance.ws_client import BinanceWSClient
from app.core.data.writer import DataWriter
from app.core.data.normalizer import DataNormalizer
from app.core.data.sequence_validator import SequenceValidator
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.strategy_manager import StrategyManager
from app.core.strategies.signal import TradingSignal
from app.core.execution.paper_exchange import PaperExchange
from app.core.risk.risk_manager import RiskManager
from app.core.risk.risk_events import RiskEvent
from app.utils.logger import get_logger
from app.config import settings
import asyncio
from typing import List, Optional

logger = get_logger("ws_manager")

class WebSocketManager:
    """Orquestador de todos los WebSocket clients"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.clients = []
        self.data_writer = DataWriter()
        self.normalizer = DataNormalizer()
        self.sequence_validator = SequenceValidator()
        self.indicator_manager = IndicatorManager()
        
        # ✨ NUEVO: Paper Exchange
        self.paper_exchange = PaperExchange(
            initial_balance=10000.0,
            exchange_name="binance",
            market_type="futures"
        )
        
        # ✨ NUEVO: Risk Manager
        self.risk_manager = RiskManager(
            initial_balance=10000.0,
            max_daily_loss_pct=2.0,
            max_weekly_loss_pct=5.0,
            max_drawdown_pct=10.0,
            max_consecutive_losses=3,
            cooldown_after_loss_minutes=15
        )
        
        # ✨ NUEVO: Strategy Manager con Paper Exchange y Risk Manager
        self.strategy_manager = StrategyManager(
            indicator_manager=self.indicator_manager,
            account_balance=10000.0,
            execution_engine=self.paper_exchange,
            risk_manager=self.risk_manager  # ✨ NUEVO
        )

        # Inicializar estrategias por defecto
        for symbol in symbols:
            # Breakout Compression Strategy
            self.strategy_manager.add_strategy(
                symbol=symbol,
                strategy_type="breakout_compression",
                enabled=True
            )
            
            # ✨ NUEVO: Mean Reversion Strategy
            self.strategy_manager.add_strategy(
                symbol=symbol,
                strategy_type="mean_reversion",
                rsi_oversold=30.0,
                rsi_overbought=70.0,
                volume_multiplier=1.2,
                atr_stop_multiplier=2.0,
                enabled=True
            )

        logger.info(
            "strategies_initialized",
            symbols=symbols,
            strategies_per_symbol=2,
            aggregation_enabled=True
        )

        # Cache para deduplicación
        self.tick_cache = set()

        # Estado
        self.is_running = False
        
        # ✨ NUEVO: Task para actualizar risk manager
        self._risk_update_task: Optional[asyncio.Task] = None

    async def start(self):
        """Iniciar todos los clientes y servicios"""
        logger.info("ws_manager_starting", symbols=self.symbols, mode=settings.MODE)

        # Crear cliente Binance Spot
        spot_client = BinanceWSClient(market_type="spot", symbols=self.symbols)

        # Registrar callbacks
        spot_client.subscribe("tick", self._handle_tick)
        spot_client.subscribe("bar", self._handle_bar)
        spot_client.subscribe("orderbook", self._handle_orderbook)
        spot_client.subscribe("max_reconnects_reached", self._handle_max_reconnects)

        self.clients.append(spot_client)

        # Crear cliente Binance Futures (si aplica)
        futures_client = BinanceWSClient(market_type="futures", symbols=self.symbols)
        futures_client.subscribe("tick", self._handle_tick)
        futures_client.subscribe("bar", self._handle_bar)
        futures_client.subscribe("orderbook", self._handle_orderbook)
        futures_client.subscribe("mark_price", self._handle_mark_price)

        self.clients.append(futures_client)

        self.is_running = True

        # ✨ NUEVO: Iniciar actualización periódica del risk manager
        self._risk_update_task = asyncio.create_task(self._update_risk_manager_loop())

        # Iniciar tasks en paralelo
        tasks = [
            client.connect() for client in self.clients
        ]
        tasks.append(self.data_writer.start_periodic_flush())

        logger.info("ws_manager_started", client_count=len(self.clients))

        await asyncio.gather(*tasks)

    async def _handle_tick(self, tick: dict):
        """Procesar tick recibido"""
        try:
            # Validar
            if not self.normalizer.validate_tick(tick):
                logger.warning("invalid_tick", tick=tick)
                return

            # Deduplicar
            if not self.normalizer.deduplicate_tick(tick, self.tick_cache):
                return

            # Escribir a DB
            await self.data_writer.add_tick(tick)

            # Distribuir tick a estrategias
            symbol = tick['symbol']
            await self.strategy_manager.on_tick(symbol, tick)

        except Exception as e:
            logger.error("handle_tick_error", error=str(e), tick=tick)

    async def _handle_bar(self, bar: dict):
        """Procesar barra recibida"""
        try:
            # Validar
            if not self.normalizer.validate_bar(bar):
                logger.warning("invalid_bar", bar=bar)
                return

            # Escribir a DB
            await self.data_writer.add_bar(bar)

            # Actualizar indicadores
            symbol = bar['symbol']
            indicator_results = self.indicator_manager.update_with_bar(symbol, bar)

            # ✨ NUEVO: Generar señal agregada
            signal = await self.strategy_manager.process_bar_and_generate_signal(symbol, bar)
            
            if signal:
                # Ejecutar señal con risk management
                await self._execute_signal(signal)
                
                logger.info(
                    "signal_executed",
                    symbol=symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    quantity=signal.quantity
                )

            logger.debug(
                "bar_processed",
                symbol=bar['symbol'],
                close=bar['close'],
                indicators_updated=len(indicator_results),
                signal_generated=signal is not None
            )

        except Exception as e:
            logger.error("handle_bar_error", error=str(e), bar=bar)

    async def _handle_orderbook(self, orderbook: dict):
        """Procesar orderbook snapshot"""
        try:
            # Validar secuencia (si tiene sequence number)
            if 'last_update_id' in orderbook:
                self.sequence_validator.validate(
                    orderbook['symbol'],
                    orderbook['last_update_id']
                )

            # Escribir a DB
            await self.data_writer.add_orderbook(orderbook)

            # Actualizar OBI
            symbol = orderbook['symbol']
            obi_results = self.indicator_manager.update_with_orderbook(symbol, orderbook)

            logger.debug(
                "orderbook_processed",
                symbol=symbol,
                bids_count=len(orderbook.get('bids', [])),
                asks_count=len(orderbook.get('asks', [])),
                obi_updated=len(obi_results)
            )

        except Exception as e:
            logger.error("handle_orderbook_error", error=str(e))

    async def _handle_mark_price(self, data: dict):
        """Procesar mark price (futures)"""
        try:
            logger.debug(
                "mark_price_received",
                symbol=data['symbol'],
                mark_price=data['mark_price'],
                funding_rate=data['funding_rate']
            )
            # TODO: Almacenar en tabla mark_prices si es necesario

        except Exception as e:
            logger.error("handle_mark_price_error", error=str(e))

    async def _handle_max_reconnects(self, data: dict):
        """Manejar fallo permanente de conexión"""
        logger.critical("max_reconnects_reached", data=data)
        # TODO: Enviar alerta crítica (email, Slack, etc.)
        # TODO: Intentar failover a otro data source

    async def stop(self):
        """Detener todos los clientes"""
        logger.info("ws_manager_stopping")
        self.is_running = False

        # ✨ NUEVO: Detener task de risk manager
        if self._risk_update_task:
            self._risk_update_task.cancel()
            try:
                await self._risk_update_task
            except asyncio.CancelledError:
                pass

        # Flush final
        await self.data_writer.flush_all()

        # Cerrar clientes
        for client in self.clients:
            await client.close()

        logger.info("ws_manager_stopped")

    async def _update_risk_manager_loop(self):
        """Loop para actualizar risk manager con balance actual"""
        while True:
            try:
                # Obtener balance actual del paper exchange
                stats = self.paper_exchange.get_statistics()
                current_balance = stats['current_balance']
                
                # Actualizar risk manager
                risk_events = self.risk_manager.update_balance(current_balance)
                
                # Procesar eventos de riesgo
                for event in risk_events:
                    await self._handle_risk_event(event)
                
                # Actualizar cada 5 segundos
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating risk manager: {str(e)}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _handle_risk_event(self, event: RiskEvent):
        """Manejar evento de riesgo"""
        logger.warning(
            "risk_event_triggered",
            type=event.type,
            severity=event.severity,
            message=event.message,
            value=event.value,
            limit=event.limit
        )
        
        # Si es crítico o emergencia, detener estrategias
        if event.should_stop_trading():
            logger.critical(
                "stopping_all_strategies",
                reason=event.message
            )
            
            # Deshabilitar todas las estrategias
            for symbol_strategies in self.strategy_manager.strategies.values():
                for strategy in symbol_strategies.values():
                    strategy.enabled = False
            
            # TODO: Broadcast event to dashboard via WebSocket
            # await dashboard_ws.broadcast({
            #     "type": "risk_event",
            #     "data": event.to_dict()
            # })

    async def _execute_signal(self, signal: TradingSignal):
        """Ejecutar señal con validación de riesgo"""
        try:
            # Validar señal con risk manager
            risk_check = await self.risk_manager.validate_signal(signal)
            
            if not risk_check.allowed:
                logger.warning(
                    "signal_rejected_by_risk",
                    symbol=signal.symbol,
                    action=signal.action,
                    reason=risk_check.reason
                )
                return
            
            # Ejecutar en paper exchange
            order_result = await self.paper_exchange.execute_signal(signal)
            
            if order_result:
                logger.info(
                    "signal_executed_successfully",
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    price=order_result.get('price'),
                    order_id=order_result.get('order_id')
                )
            else:
                logger.error("signal_execution_failed", symbol=signal.symbol)
                
        except Exception as e:
            logger.error(
                "signal_execution_error",
                symbol=signal.symbol,
                error=str(e)
            )

    async def get_status(self) -> dict:
        """Obtener estado del manager"""
        return {
            "is_running": self.is_running,
            "clients": [
                {
                    "market_type": client.market_type,
                    "is_connected": client.is_connected,
                    "reconnect_attempts": client.reconnect_attempts,
                    "symbols": client.symbols
                }
                for client in self.clients
            ],
            "writer_stats": await self.data_writer.get_stats(),
            "sequence_gaps": self.sequence_validator.gaps_detected,
            "indicators": {
                "active_symbols": list(self.indicator_manager.indicators.keys()),
                "cache_connected": self.indicator_manager.cache.redis is not None
            },
            # ✨ NUEVO: Estado del Paper Exchange
            "paper_exchange": self.paper_exchange.get_statistics(),
            # ✨ NUEVO: Estado del Risk Manager
            "risk_manager": self.risk_manager.get_statistics() if self.risk_manager else None,
            "strategies": self.strategy_manager.get_statistics()
        }

# Instancia global del WebSocket Manager
ws_manager = WebSocketManager(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])