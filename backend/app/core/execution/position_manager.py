# backend/app/core/execution/position_manager.py

from typing import Optional, Dict, Any
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger("position_manager")

class PositionManager:
    """
    Gestión de posiciones con TP/SL/Trailing
    """
    
    def __init__(self):
        self.open_positions: Dict[str, Dict] = {}
        self.closed_positions: list = []
    
    def open_position(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        timestamp: int,
        strategy: str
    ) -> Dict:
        """Abrir nueva posición"""
        
        position = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_timestamp": timestamp,
            "strategy": strategy,
            "trailing_active": False,
            "trailing_peak": entry_price,
            "status": "OPEN"
        }
        
        self.open_positions[trade_id] = position
        
        logger.info(
            "position_opened",
            trade_id=trade_id,
            side=side,
            entry=entry_price,
            sl=stop_loss,
            tp=take_profit
        )
        
        return position
    
    def check_exits(
        self,
        trade_id: str,
        current_bar: Dict,
        trailing_config: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Verificar si debe cerrar posición (TP/SL/Trailing)
        
        Returns:
            Dict con exit order si debe cerrar, None si mantener
        """
        
        if trade_id not in self.open_positions:
            return None
        
        position = self.open_positions[trade_id]
        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        # Calcular PnL %
        if position['side'] == "BUY":
            pnl_pct = ((current_price - position['entry_price']) / 
                      position['entry_price']) * 100
        else:  # SELL
            pnl_pct = ((position['entry_price'] - current_price) / 
                      position['entry_price']) * 100
        
        # 1. CHECK STOP LOSS (prioridad máxima)
        if position['side'] == "BUY":
            if low <= position['stop_loss']:
                return self._create_exit_order(
                    position,
                    exit_price=position['stop_loss'],
                    reason="stop_loss",
                    order_type="MARKET"
                )
        else:  # SELL
            if high >= position['stop_loss']:
                return self._create_exit_order(
                    position,
                    exit_price=position['stop_loss'],
                    reason="stop_loss",
                    order_type="MARKET"
                )
        
        # 2. CHECK TAKE PROFIT
        if position['side'] == "BUY":
            if high >= position['take_profit']:
                return self._create_exit_order(
                    position,
                    exit_price=position['take_profit'],
                    reason="take_profit",
                    order_type="LIMIT"
                )
        else:  # SELL
            if low <= position['take_profit']:
                return self._create_exit_order(
                    position,
                    exit_price=position['take_profit'],
                    reason="take_profit",
                    order_type="LIMIT"
                )
        
        # 3. CHECK TRAILING STOP (si configurado)
        if trailing_config:
            trailing_exit = self._check_trailing_stop(
                position,
                current_price,
                high,
                low,
                trailing_config
            )
            if trailing_exit:
                return trailing_exit
        
        return None  # Mantener posición
    
    def _check_trailing_stop(
        self,
        position: Dict,
        current_price: float,
        high: float,
        low: float,
        config: Dict
    ) -> Optional[Dict]:
        """Verificar trailing stop"""
        
        activation_pct = config.get('activation_pct', 0.2)
        distance_pct = config.get('distance_pct', 0.1)
        
        entry_price = position['entry_price']
        
        # Calcular profit actual
        if position['side'] == "BUY":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Activar trailing si alcanzó activation threshold
        if profit_pct >= activation_pct:
            if not position['trailing_active']:
                position['trailing_active'] = True
                position['trailing_peak'] = current_price
                logger.info(
                    "trailing_activated",
                    trade_id=position['trade_id'],
                    profit_pct=profit_pct
                )
            
            # Actualizar peak
            if position['side'] == "BUY":
                if high > position['trailing_peak']:
                    position['trailing_peak'] = high
                
                # Check si cayó distance_pct desde peak
                drop = ((position['trailing_peak'] - low) / 
                       position['trailing_peak']) * 100
                
                if drop >= distance_pct:
                    exit_price = position['trailing_peak'] * (1 - distance_pct / 100)
                    return self._create_exit_order(
                        position,
                        exit_price=exit_price,
                        reason="trailing_stop",
                        order_type="MARKET"
                    )
            
            else:  # SELL
                if low < position['trailing_peak']:
                    position['trailing_peak'] = low
                
                rise = ((high - position['trailing_peak']) / 
                       position['trailing_peak']) * 100
                
                if rise >= distance_pct:
                    exit_price = position['trailing_peak'] * (1 + distance_pct / 100)
                    return self._create_exit_order(
                        position,
                        exit_price=exit_price,
                        reason="trailing_stop",
                        order_type="MARKET"
                    )
        
        return None
    
    def _create_exit_order(
        self,
        position: Dict,
        exit_price: float,
        reason: str,
        order_type: str
    ) -> Dict:
        """Crear orden de salida"""
        
        exit_order = {
            "trade_id": position['trade_id'],
            "action": "CLOSE",
            "side": "SELL" if position['side'] == "BUY" else "BUY",
            "type": order_type,
            "price": exit_price,
            "quantity": position['quantity'],
            "reason": reason
        }
        
        # Mover a closed
        position['status'] = "CLOSED"
        position['exit_reason'] = reason
        position['exit_price'] = exit_price
        
        self.closed_positions.append(position)
        del self.open_positions[position['trade_id']]
        
        logger.info(
            "position_closed",
            trade_id=position['trade_id'],
            reason=reason,
            exit_price=exit_price
        )
        
        return exit_order
    
    def get_open_positions(self) -> Dict[str, Dict]:
        """Obtener todas las posiciones abiertas"""
        return self.open_positions
    
    def get_position(self, trade_id: str) -> Optional[Dict]:
        """Obtener posición específica"""
        return self.open_positions.get(trade_id)
