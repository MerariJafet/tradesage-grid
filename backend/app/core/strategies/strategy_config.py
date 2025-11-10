from typing import Dict, Any

class StrategyConfig:
    """Configuraciones predefinidas para estrategias"""

    # Breakout Compression Strategy
    BREAKOUT_COMPRESSION = {
        "bb_compression_threshold": 0.02,  # 2% bandwidth
        "volume_multiplier": 1.2,  # 120% del promedio
        "rsi_min": 30,
        "rsi_max": 70,
        "stop_atr_multiplier": 0.75,
        "tp_atr_multiplier": 1.2,
        "min_confidence": 0.60
    }

    # Mean Reversion Strategy - Conservative
    MEAN_REVERSION_CONSERVATIVE = {
        "rsi_oversold": 25,  # Más extremo
        "rsi_overbought": 75,
        "volume_multiplier": 1.5,  # Más volumen requerido
        "atr_stop_multiplier": 2.5  # Stop más amplio
    }

    # Mean Reversion Strategy - Aggressive
    MEAN_REVERSION_AGGRESSIVE = {
        "rsi_oversold": 35,  # Menos extremo (más señales)
        "rsi_overbought": 65,
        "volume_multiplier": 1.2,  # Menos volumen requerido
        "atr_stop_multiplier": 1.5  # Stop más ajustado
    }

    # Mean Reversion Strategy - Balanced (Default)
    MEAN_REVERSION_BALANCED = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "volume_multiplier": 1.2,
        "atr_stop_multiplier": 2.0
    }

    @classmethod
    def get_strategy_config(cls, strategy_type: str, profile: str = "balanced") -> Dict[str, Any]:
        """
        Obtener configuración de estrategia

        Args:
            strategy_type: "breakout_compression" o "mean_reversion"
            profile: "conservative", "balanced", "aggressive"
        """
        if strategy_type == "breakout_compression":
            return cls.BREAKOUT_COMPRESSION.copy()

        elif strategy_type == "mean_reversion":
            if profile == "conservative":
                return cls.MEAN_REVERSION_CONSERVATIVE.copy()
            elif profile == "aggressive":
                return cls.MEAN_REVERSION_AGGRESSIVE.copy()
            else:  # balanced
                return cls.MEAN_REVERSION_BALANCED.copy()

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")