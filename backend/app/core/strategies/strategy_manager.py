from typing import Dict, List, Optional
from app.core.strategies.base import BaseStrategy
from app.core.strategies.breakout_compression import BreakoutCompressionStrategy
from app.core.strategies.mean_reversion import MeanReversionStrategy
from app.core.strategies.signal_aggregator import SignalAggregatorV2
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.utils.logger import get_logger

logger = get_logger("strategy_manager")

class StrategyManager:
    """
    Manager central de estrategias
    - Crea y gestiona múltiples estrategias
    - Distribuye eventos de mercado a estrategias activas
    - Recopila estadísticas
    """

    def __init__(
        self,
        indicator_manager: IndicatorManager,
        account_balance: float = 10000.0,  # Default para paper trading
        execution_engine = None,  # ✨ NUEVO
        risk_manager = None  # ✨ NUEVO
    ):
        self.indicator_manager = indicator_manager
        self.account_balance = account_balance
        self.execution_engine = execution_engine  # ✨ NUEVO
        self.risk_manager = risk_manager  # ✨ NUEVO

        # Componentes compartidos
        self.position_sizer = PositionSizer(account_balance)
        self.signal_validator = SignalValidator(
            indicator_manager=indicator_manager,
            max_open_positions=3,
            max_daily_trades=20
        )
        
        # ✨ NUEVO: Signal Aggregator
        self.signal_aggregator = SignalAggregatorV2()

        # Estrategias por símbolo
        # estructura: {symbol: {strategy_name: strategy_instance}}
        self.strategies: Dict[str, Dict[str, BaseStrategy]] = {}

    def add_strategy(
        self,
        symbol: str,
        strategy_type: str,
        **strategy_params
    ) -> BaseStrategy:
        """
        Añadir estrategia para un símbolo

        Args:
            symbol: Símbolo (ej: BTCUSDT)
            strategy_type: Tipo de estrategia ('breakout_compression', etc.)
            **strategy_params: Parámetros específicos de la estrategia

        Returns:
            Instancia de la estrategia creada
        """

        if symbol not in self.strategies:
            self.strategies[symbol] = {}

        # Crear estrategia según tipo
        if strategy_type == "breakout_compression":
            strategy = BreakoutCompressionStrategy(
                symbol=symbol,
                indicator_manager=self.indicator_manager,
                position_sizer=self.position_sizer,
                signal_validator=self.signal_validator,
                execution_engine=self.execution_engine,  # ✨ NUEVO
                risk_manager=self.risk_manager,  # ✨ NUEVO
                **strategy_params
            )
        elif strategy_type == "mean_reversion":
            strategy = MeanReversionStrategy(
                symbol=symbol,
                indicator_manager=self.indicator_manager,
                position_sizer=self.position_sizer,
                signal_validator=self.signal_validator,
                execution_engine=self.execution_engine,
                risk_manager=self.risk_manager,
                **strategy_params
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Registrar
        self.strategies[symbol][strategy.name] = strategy

        logger.info(
            "strategy_added",
            symbol=symbol,
            strategy_name=strategy.name,
            strategy_type=strategy_type,
            has_execution_engine=self.execution_engine is not None,
            has_risk_manager=self.risk_manager is not None  # ✨ NUEVO
        )

        return strategy

    async def on_bar(self, symbol: str, bar: Dict):
        """Distribuir evento de barra a estrategias del símbolo"""
        if symbol not in self.strategies:
            return

        for strategy in self.strategies[symbol].values():
            if strategy.enabled:
                await strategy.on_bar(bar)

    async def process_bar_and_generate_signal(self, symbol: str, bar: Dict):
        """
        Procesar barra y generar señal agregada si hay múltiples estrategias
        
        Returns:
            TradingSignal agregada o None
        """
        if symbol not in self.strategies:
            return None

        # Recolectar señales de todas las estrategias activas
        signals = []
        
        for strategy_name, strategy in self.strategies[symbol].items():
            if not strategy.enabled:
                continue
            
            # Generar señal usando el método check_setup (síncrono)
            if hasattr(strategy, 'check_setup'):
                signal = strategy.check_setup(bar)
                if signal:
                    signals.append(signal)
                    logger.info(
                        "strategy_signal_generated",
                        strategy=strategy_name,
                        symbol=symbol,
                        action=signal.action,
                        confidence=signal.confidence
                    )

        # Si hay múltiples señales, usar aggregator
        if len(signals) > 1:
            aggregated_signal = self.signal_aggregator.get_aggregated_signal_from_list(symbol, signals)
            
            if aggregated_signal:
                logger.info(
                    "aggregated_signal_generated",
                    symbol=symbol,
                    action=aggregated_signal.action,
                    confidence=aggregated_signal.confidence,
                    source_strategies=len(signals)
                )
                return aggregated_signal
        
        # Si solo hay una señal, retornarla directamente
        elif len(signals) == 1:
            return signals[0]
        
        return None

    async def on_tick(self, symbol: str, tick: Dict):
        """Distribuir evento de tick a estrategias del símbolo"""
        if symbol not in self.strategies:
            return

        for strategy in self.strategies[symbol].values():
            if strategy.enabled:
                await strategy.on_tick(tick)

    async def on_orderbook(self, symbol: str, orderbook: Dict):
        """Distribuir evento de orderbook a estrategias del símbolo"""
        if symbol not in self.strategies:
            return

        for strategy in self.strategies[symbol].values():
            if strategy.enabled:
                await strategy.on_orderbook(orderbook)

    def get_strategy(self, symbol: str, strategy_name: str) -> Optional[BaseStrategy]:
        """Obtener estrategia específica"""
        if symbol not in self.strategies:
            return None
        return self.strategies[symbol].get(strategy_name)

    def get_all_strategies(self, symbol: Optional[str] = None) -> List[BaseStrategy]:
        """Obtener todas las estrategias (opcionalmente filtradas por símbolo)"""
        all_strategies = []

        if symbol:
            if symbol in self.strategies:
                all_strategies.extend(self.strategies[symbol].values())
        else:
            for symbol_strategies in self.strategies.values():
                all_strategies.extend(symbol_strategies.values())

        return all_strategies

    def get_statistics(self, symbol: Optional[str] = None) -> Dict:
        """Obtener estadísticas de todas las estrategias"""
        stats = {}

        strategies = self.get_all_strategies(symbol)

        for strategy in strategies:
            key = f"{strategy.symbol}_{strategy.name}"
            stats[key] = strategy.get_statistics()

        # Agregar estadísticas globales
        stats['global'] = {
            'total_strategies': len(strategies),
            'enabled_strategies': sum(1 for s in strategies if s.enabled),
            'total_open_positions': sum(1 for s in strategies if s.open_position),
            'account_balance': self.account_balance
        }

        return stats

    def enable_strategy(self, symbol: str, strategy_name: str):
        """Habilitar estrategia"""
        strategy = self.get_strategy(symbol, strategy_name)
        if strategy:
            strategy.enabled = True
            logger.info("strategy_enabled", symbol=symbol, strategy=strategy_name)

    def disable_strategy(self, symbol: str, strategy_name: str):
        """Deshabilitar estrategia"""
        strategy = self.get_strategy(symbol, strategy_name)
        if strategy:
            strategy.enabled = False
            logger.info("strategy_disabled", symbol=symbol, strategy=strategy_name)

    def update_account_balance(self, new_balance: float):
        """Actualizar balance de cuenta"""
        self.account_balance = new_balance
        self.position_sizer.update_account_balance(new_balance)
        logger.info("account_balance_updated", new_balance=new_balance)