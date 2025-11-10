# backend/app/core/strategies/hybrid_momentum_meanrev.py

"""
Hybrid Momentum + Mean Reversion Strategy

Filosofía:
- Indicadores técnicos tradicionales como SEÑAL PRIMARIA (70%)
- Order book features como FILTROS de calidad (30%)
- Sin time limits - dejar que el mercado decida
- TP/SL ratios realistas para el mercado

Señales:
1. LONG: RSI oversold + BB touch + Volume surge + OB filter
2. SHORT: RSI overbought + BB touch + Volume surge + OB filter

Exits:
- TP: 0.50% (más realista)
- SL: 0.20% (ratio 2.5:1)
- Trailing: Activar a 0.35%, trail 0.12%
- NO time limit
"""

from typing import Optional, Dict
from app.core.strategies.base import BaseStrategy
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.utils.logger import get_logger

logger = get_logger("hybrid_strategy")

class HybridMomentumMeanRevStrategy(BaseStrategy):
    """
    Estrategia híbrida: Indicadores técnicos + Order Book filters
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
            name="HybridMomentumMeanRev",
            symbol=symbol,
            indicator_manager=indicator_manager,
            position_sizer=position_sizer,
            signal_validator=signal_validator
        )
        
        # Configuración por defecto
        self.config = {
            # 📊 SEÑALES PRIMARIAS (Indicadores Técnicos)
            "rsi_period": 7,
            "rsi_oversold": 25,          # < 25 = oversold
            "rsi_overbought": 75,        # > 75 = overbought
            
            "bb_period": 15,
            "bb_std": 2.0,
            "bb_touch_threshold": 0.005,  # Dentro de 0.5% de la banda
            
            "macd_fast": 8,
            "macd_slow": 17,
            "macd_signal": 9,
            
            "volume_multiplier": 0,       # ❌ DESACTIVADO - orderbook snapshots no tienen volumen real
            
            # 🔍 FILTROS DE CONFIRMACIÓN (Solo técnicos)
            "adx_period": 14,
            "adx_min": 15,                # ✅ RELAJADO: 20 → 15 (V2.2.2)
            
            # 📈 FILTROS ORDER BOOK - ❌ ELIMINADOS EN V2.2.2
            # All OrderBook filters removed - pure technical strategy
            
            # 💰 EXIT PARAMETERS (Sin time limit) - Ajustado para 5-min
            "tp_pct": 0.60,               # Take profit 0.6% (5-min adjusted)
            "sl_pct": 0.25,               # Stop loss 0.25% (ratio 2.4:1)
            "trailing_activation_pct": 0.35,  # Activar trailing a +0.35%
            "trailing_distance_pct": 0.12,    # Trail 0.12% atrás
            "use_time_limit": False,          # NO usar time limit
            
            # 🛡️ RISK
            "risk_per_trade_pct": 0.8,
            "max_open_positions": 3,
        }
        
        if config:
            self.config.update(config)
        
        # Contadores de diagnóstico
        self.total_bars_processed = 0
        self.signals_rsi = 0
        self.signals_bb = 0
        self.signals_macd = 0
        self.filtered_by_volume = 0
        self.filtered_by_adx = 0
        self.filtered_by_orderbook = 0
        self.signals_generated = 0
        
        logger.info(
            "hybrid_strategy_initialized",
            symbol=symbol,
            config_summary={
                "rsi_levels": f"{self.config['rsi_oversold']}/{self.config['rsi_overbought']}",
                "bb_period": self.config['bb_period'],
                "tp_sl": f"{self.config['tp_pct']}%/{self.config['sl_pct']}%",
                "time_limit": self.config['use_time_limit']
            }
        )
    
    def check_setup(self, bar: Dict) -> Optional[TradingSignal]:
        """
        Verificar señales con lógica híbrida
        """
        
        self.total_bars_processed += 1
        
        # Extraer datos
        price = bar['close']
        
        # Indicadores técnicos
        rsi = bar.get('rsi_7', 50)
        bb_upper = bar.get('bb_upper', price * 1.02)
        bb_lower = bar.get('bb_lower', price * 0.98)
        bb_middle = bar.get('bb_middle', price)
        macd = bar.get('macd', 0)
        macd_signal = bar.get('macd_signal', 0)
        macd_hist = bar.get('macd_histogram', 0)
        volume = bar.get('volume', 0)
        volume_sma = bar.get('volume_sma_20', 1)
        adx = bar.get('adx_14', 0)
        
        # Order book features
        depth_imb = bar.get('DepthImb_L2L5', 0)
        spread_bps = bar.get('Spread_bps', 999)
        volatility = bar.get('VolatilityScore', 0)
        
        # ==========================================
        # PARTE 1: SEÑALES PRIMARIAS (Técnicas)
        # ==========================================
        
        # 🟢 LONG SIGNAL CONDITIONS
        long_rsi = rsi < self.config['rsi_oversold']
        long_bb = price <= bb_lower * (1 + self.config['bb_touch_threshold'])
        long_macd = macd_hist > 0 and macd > macd_signal
        
        # 🔴 SHORT SIGNAL CONDITIONS  
        short_rsi = rsi > self.config['rsi_overbought']
        short_bb = price >= bb_upper * (1 - self.config['bb_touch_threshold'])
        short_macd = macd_hist < 0 and macd < macd_signal
        
        # Contadores de señales
        if long_rsi or short_rsi:
            self.signals_rsi += 1
        if long_bb or short_bb:
            self.signals_bb += 1
        if long_macd or short_macd:
            self.signals_macd += 1
        
        # ==========================================
        # PARTE 2: FILTROS DE CONFIRMACIÓN
        # ==========================================
        
        # Filtro de volumen (DESACTIVADO si multiplier = 0)
        if self.config['volume_multiplier'] > 0:
            volume_surge = volume > volume_sma * self.config['volume_multiplier']
            if not volume_surge:
                if long_rsi or short_rsi or long_bb or short_bb:
                    self.filtered_by_volume += 1
                return None
        
        # Filtro de tendencia (ADX)
        trend_exists = adx > self.config['adx_min']
        if not trend_exists:
            if long_rsi or short_rsi or long_bb or short_bb:
                self.filtered_by_adx += 1
            return None
        
        # ==========================================
        # PARTE 3: FILTROS ORDER BOOK - ❌ DISABLED V2.2.2
        # ==========================================
        
        # OrderBook filters REMOVED based on V2.2.1 analysis:
        # - DepthImb blocked 64% of signals even at ±0.01
        # - No clear value added vs pure technical indicators
        # - Strategy now relies on RSI + BB + ADX + Volume only
        
        # All OB filtering code REMOVED
        
        # ==========================================
        # PARTE 4: GENERAR SEÑALES
        # ==========================================
        
        # 🟢 LONG SETUP
        if (long_rsi or long_bb) and trend_exists:
            
            # ❌ NO OB CONFIRMATION NEEDED (V2.2.2)
            # Proceed directly to trade generation
            
            # Calcular stops
            stop_loss = price - (price * self.config['sl_pct'] / 100)
            take_profit = price + (price * self.config['tp_pct'] / 100)
            
            # Position sizing
            quantity = self.position_sizer.calculate_position_size(
                symbol=self.symbol,
                entry_price=price,
                stop_loss=stop_loss
            )
            
            self.signals_generated += 1
            
            # Determinar razón principal
            if long_rsi and long_bb:
                reason = "RSI_BB_OVERSOLD"
            elif long_rsi:
                reason = "RSI_OVERSOLD"
            else:
                reason = "BB_LOWER_TOUCH"
            
            signal = TradingSignal(
                strategy_name=self.name,
                symbol=self.symbol,
                signal_type=SignalType.MOMENTUM,
                action=SignalAction.BUY,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                confidence=0.75,  # Fixed confidence, no OB dependency
                reason=f"{reason}+ADX (RSI:{rsi:.1f}, ADX:{adx:.1f})",
                indicators=bar
            )
            
            logger.info(
                "long_signal",
                reason=reason,
                rsi=rsi,
                adx=adx,
                price=price
            )
            
            return signal
        
        # 🔴 SHORT SETUP
        elif (short_rsi or short_bb) and trend_exists:
            
            # ❌ NO OB CONFIRMATION NEEDED (V2.2.2)
            # Proceed directly to trade generation
            
            stop_loss = price + (price * self.config['sl_pct'] / 100)
            take_profit = price - (price * self.config['tp_pct'] / 100)
            
            quantity = self.position_sizer.calculate_position_size(
                symbol=self.symbol,
                entry_price=price,
                stop_loss=stop_loss
            )
            
            self.signals_generated += 1
            
            if short_rsi and short_bb:
                reason = "RSI_BB_OVERBOUGHT"
            elif short_rsi:
                reason = "RSI_OVERBOUGHT"
            else:
                reason = "BB_UPPER_TOUCH"
            
            signal = TradingSignal(
                strategy_name=self.name,
                symbol=self.symbol,
                signal_type=SignalType.MOMENTUM,
                action=SignalAction.SELL,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                confidence=0.75,  # Fixed confidence, no OB dependency
                reason=f"{reason}+ADX (RSI:{rsi:.1f}, ADX:{adx:.1f})",
                indicators=bar
            )
            
            logger.info(
                "short_signal",
                reason=reason,
                rsi=rsi,
                adx=adx,
                price=price
            )
            
            return signal
        
        return None
    
    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Implementación requerida por BaseStrategy
        Delega a check_setup
        """
        bar = market_data.get('bar', {})
        return self.check_setup(bar)
    
    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calcular stop loss basado en porcentaje configurado"""
        sl_pct = self.config['sl_pct'] / 100
        if side == "BUY":
            return entry_price * (1 - sl_pct)
        else:  # SELL
            return entry_price * (1 + sl_pct)
    
    def get_take_profit(self, entry_price: float, side: str, atr: float) -> float:
        """Calcular take profit basado en porcentaje configurado"""
        tp_pct = self.config['tp_pct'] / 100
        if side == "BUY":
            return entry_price * (1 + tp_pct)
        else:  # SELL
            return entry_price * (1 - tp_pct)
    
    def get_diagnostics(self) -> Dict:
        """Obtener estadísticas de diagnóstico"""
        
        total_potential = self.signals_rsi + self.signals_bb + self.signals_macd
        
        return {
            "bars_processed": self.total_bars_processed,
            "potential_signals": {
                "rsi": self.signals_rsi,
                "bb": self.signals_bb,
                "macd": self.signals_macd,
                "total": total_potential
            },
            "filtered_by": {
                "volume": self.filtered_by_volume,
                "adx": self.filtered_by_adx,
                "orderbook": self.filtered_by_orderbook,
                "total": self.filtered_by_volume + self.filtered_by_adx + self.filtered_by_orderbook
            },
            "signals_generated": self.signals_generated,
            "conversion_rate": (self.signals_generated / total_potential * 100) if total_potential > 0 else 0
        }
