# backend/app/core/strategies/orderbook_imbalance.py

from typing import Optional, Dict
from app.core.strategies.base import BaseStrategy
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.utils.logger import get_logger

logger = get_logger("orderbook_imbalance_strategy")

class OrderBookImbalanceStrategy(BaseStrategy):
    """
    Scalping basado en desequilibrio del order book (Level 2-5)
    
    📊 Concepto:
    El feature DepthImb_L2L5 mide el desequilibrio entre bid y ask en las
    capas 2-5 del order book. Correlación validada: 0.2874 con returns
    
    📈 Lógica:
    - DepthImb_L2L5 > +0.15 → Presión compradora fuerte → LONG
    - DepthImb_L2L5 < -0.15 → Presión vendedora fuerte → SHORT
    
    🎯 Features usados:
    - DepthImb_L2L5: Desequilibrio bid/ask (principal)
    - spread: Spread del order book (filtro)
    - TradeIntensity_1s: Intensidad de trades (confirmación)
    - KyleLambda_1s: Impact de precio (filtro)
    
    ⚙️ Exits:
    - TP: 0.30% (tight para scalping)
    - SL: 0.15% (muy tight)
    - Trailing: Activar a +0.20%, trail 0.08%
    - Time: 5 min máximo
    """
    
    def __init__(
        self,
        symbol: str,
        indicator_manager,
        position_sizer,
        signal_validator,
        config: Optional[Dict] = None
    ):
        super().__init__(
            name="OrderBookImbalance",
            symbol=symbol,
            indicator_manager=indicator_manager,
            position_sizer=position_sizer,
            signal_validator=signal_validator
        )
        
        # Configuración por defecto
        self.config = {
            # 🎯 Entry thresholds
            "imbalance_long_threshold": 0.15,     # DepthImb > 0.15 = Strong buy pressure
            "imbalance_short_threshold": -0.15,   # DepthImb < -0.15 = Strong sell pressure
            "min_trade_intensity": 0.5,           # TradeIntensity mínima
            "max_spread": 1.0,                    # Spread < 1.0 (tight spread)
            "max_kyle_lambda": 0.5,               # KyleLambda para filtrar
            
            # 💰 Exit parameters
            "tp_pct": 0.30,                       # Take profit 0.3%
            "sl_pct": 0.15,                       # Stop loss 0.15%
            "trailing_activation_pct": 0.20,     # Activar trailing a +0.2%
            "trailing_distance_pct": 0.08,       # Trail 0.08% atrás del peak
            "max_hold_time_seconds": 300,        # 5 minutos max hold
            
            # 🛡️ Risk management
            "risk_per_trade_pct": 0.8,           # Arriesgar 0.8% por trade
            "max_open_positions": 3,             # Máximo 3 posiciones simultáneas
        }
        
        # Override con config personalizada si existe
        if config:
            self.config.update(config)
        
        # 📊 CONTADORES DE FILTRADO (para diagnóstico)
        self.filter_stats = {
            "total_bars": 0,
            "potential_long_signals": 0,      # DepthImb > threshold
            "potential_short_signals": 0,     # DepthImb < -threshold
            "filtered_by_spread": 0,
            "filtered_by_trade_intensity": 0,
            "filtered_by_kyle_lambda": 0,
            "passed_all_filters": 0,
            "actual_signals_generated": 0,
        }
        
        logger.info(
            "orderbook_strategy_initialized",
            symbol=symbol,
            imbalance_thresholds=f"+{self.config['imbalance_long_threshold']}/-{abs(self.config['imbalance_short_threshold'])}",
            tp_sl=f"{self.config['tp_pct']}%/{self.config['sl_pct']}%"
        )
    
    def check_setup(self, bar: Dict) -> Optional[TradingSignal]:
        """
        Verificar si hay setup de entrada basado en order book imbalance
        
        Args:
            bar: Diccionario con datos de la barra + features
            
        Returns:
            TradingSignal si hay entrada válida, None si no
        """
        
        # 📊 Contador de barras totales
        self.filter_stats["total_bars"] += 1
        
        # 📊 Extraer features del bar
        depth_imb = bar.get('DepthImb_L2L5', 0)
        spread = bar.get('spread', 999)
        trade_intensity = bar.get('TradeIntensity_1s', 0)
        kyle_lambda = bar.get('KyleLambda_1s', 999)
        price = bar['close']
        
        # � Contar señales potenciales (ANTES de filtros)
        is_potential_long = depth_imb > self.config['imbalance_long_threshold']
        is_potential_short = depth_imb < self.config['imbalance_short_threshold']
        
        if is_potential_long:
            self.filter_stats["potential_long_signals"] += 1
        elif is_potential_short:
            self.filter_stats["potential_short_signals"] += 1
        
        # Si no hay señal potencial, salir temprano
        if not (is_potential_long or is_potential_short):
            return None
        
        # �🚫 Filtros de calidad (CONTAR cada uno)
        
        # 1. Spread demasiado amplio = mala ejecución
        if spread > self.config['max_spread']:
            self.filter_stats["filtered_by_spread"] += 1
            return None
        
        # 2. Trade intensity baja
        if trade_intensity < self.config['min_trade_intensity']:
            self.filter_stats["filtered_by_trade_intensity"] += 1
            return None  # Poca actividad, evitar
        
        # 3. Kyle Lambda muy alto = mercado con alto impacto
        if abs(kyle_lambda) > self.config['max_kyle_lambda']:
            self.filter_stats["filtered_by_kyle_lambda"] += 1
            return None
        
        # 📊 Si llegamos aquí, pasó todos los filtros
        self.filter_stats["passed_all_filters"] += 1
        
        # 🟢 LONG SIGNAL: Presión compradora fuerte
        if is_potential_long:
            
            # Calcular stops basados en volatilidad
            atr = bar.get('atr_7', 0)
            if atr == 0:
                # Fallback si no hay ATR: usar porcentaje fijo
                atr = price * 0.002  # 0.2% como ATR estimado
            
            stop_loss = price - (price * self.config['sl_pct'] / 100)
            take_profit = price + (price * self.config['tp_pct'] / 100)
            
            # Calcular tamaño de posición
            quantity = self.position_sizer.calculate_position_size(
                symbol=self.symbol,
                entry_price=price,
                stop_loss=stop_loss,
                atr=atr
            )
            
            # Crear señal
            signal = TradingSignal(
                strategy_name=self.name,
                symbol=self.symbol,
                signal_type=SignalType.MOMENTUM,
                action=SignalAction.BUY,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                confidence=min(abs(depth_imb), 1.0),  # Confidence = fuerza del imbalance
                reason=f"OB_LONG: DepthImb={depth_imb:.3f}, Spread={spread:.3f}, TI={trade_intensity:.2f}",
                indicators=bar
            )
            
            # 📊 Contador de señales generadas
            self.filter_stats["actual_signals_generated"] += 1
            
            logger.debug(
                "long_signal_generated",
                depth_imb=depth_imb,
                price=price,
                sl=stop_loss,
                tp=take_profit
            )
            
            return signal
        
        # 🔴 SHORT SIGNAL: Presión vendedora fuerte
        elif depth_imb < self.config['imbalance_short_threshold']:
            
            atr = bar.get('atr_7', 0)
            if atr == 0:
                atr = price * 0.002
            
            stop_loss = price + (price * self.config['sl_pct'] / 100)
            take_profit = price - (price * self.config['tp_pct'] / 100)
            
            quantity = self.position_sizer.calculate_position_size(
                symbol=self.symbol,
                entry_price=price,
                stop_loss=stop_loss,
                atr=atr
            )
            
            signal = TradingSignal(
                strategy_name=self.name,
                symbol=self.symbol,
                signal_type=SignalType.MOMENTUM,
                action=SignalAction.SELL,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                confidence=min(abs(depth_imb), 1.0),
                reason=f"OB_SHORT: DepthImb={depth_imb:.3f}, Spread={spread:.3f}, TI={trade_intensity:.2f}",
                indicators=bar
            )
            
            # 📊 Contador de señales generadas
            self.filter_stats["actual_signals_generated"] += 1
            
            logger.debug(
                "short_signal_generated",
                depth_imb=depth_imb,
                price=price,
                sl=stop_loss,
                tp=take_profit
            )
            
            return signal
        
        # ⏸️ No signal: Imbalance no suficientemente fuerte
        return None
    
    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Implementación requerida por BaseStrategy
        Delega a check_setup
        """
        bar = market_data.get('bar', {})
        return self.check_setup(bar)
    
    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """
        Calcular stop loss basado en porcentaje configurado
        """
        sl_pct = self.config['sl_pct'] / 100
        if side == "BUY":
            return entry_price * (1 - sl_pct)
        else:  # SELL
            return entry_price * (1 + sl_pct)
    
    def get_take_profit(self, entry_price: float, side: str, atr: float) -> float:
        """
        Calcular take profit basado en porcentaje configurado
        """
        tp_pct = self.config['tp_pct'] / 100
        if side == "BUY":
            return entry_price * (1 + tp_pct)
        else:  # SELL
            return entry_price * (1 - tp_pct)
    
    def get_config_summary(self) -> Dict:
        """Obtener resumen de configuración"""
        return {
            "strategy": self.name,
            "thresholds": {
                "long": self.config['imbalance_long_threshold'],
                "short": self.config['imbalance_short_threshold']
            },
            "exits": {
                "tp": f"{self.config['tp_pct']}%",
                "sl": f"{self.config['sl_pct']}%",
                "trailing": f"{self.config['trailing_distance_pct']}%"
            },
            "filters": {
                "max_spread": self.config['max_spread'],
                "min_trade_intensity": self.config['min_trade_intensity']
            }
        }
    
    def get_filter_stats(self) -> Dict:
        """
        Obtener estadísticas de filtrado para diagnóstico
        
        Returns:
            Dict con contadores y porcentajes de filtrado
        """
        stats = self.filter_stats.copy()
        
        # Calcular porcentajes
        total_potential = stats["potential_long_signals"] + stats["potential_short_signals"]
        
        if total_potential > 0:
            stats["filtered_by_spread_pct"] = (stats["filtered_by_spread"] / total_potential) * 100
            stats["filtered_by_trade_intensity_pct"] = (stats["filtered_by_trade_intensity"] / total_potential) * 100
            stats["filtered_by_kyle_lambda_pct"] = (stats["filtered_by_kyle_lambda"] / total_potential) * 100
            stats["passed_all_filters_pct"] = (stats["passed_all_filters"] / total_potential) * 100
            stats["conversion_rate"] = (stats["actual_signals_generated"] / total_potential) * 100
        else:
            stats["filtered_by_spread_pct"] = 0
            stats["filtered_by_trade_intensity_pct"] = 0
            stats["filtered_by_kyle_lambda_pct"] = 0
            stats["passed_all_filters_pct"] = 0
            stats["conversion_rate"] = 0
        
        return stats
