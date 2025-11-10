from collections import defaultdict
from app.utils.logger import get_logger

logger = get_logger("sequence_validator")

class SequenceValidator:
    """Detecta gaps en secuencias de mensajes"""

    def __init__(self):
        # Último sequence number visto por símbolo
        self.last_sequences: defaultdict[str, int] = defaultdict(int)
        self.gaps_detected = 0

    def validate(self, symbol: str, sequence: int) -> bool:
        """
        Validar que no haya gaps en la secuencia
        Returns: True si OK, False si hay gap
        """
        last_seq = self.last_sequences[symbol]

        if last_seq == 0:
            # Primera vez, inicializar
            self.last_sequences[symbol] = sequence
            return True

        expected = last_seq + 1

        if sequence != expected:
            gap_size = sequence - expected
            logger.warning(
                "sequence_gap_detected",
                symbol=symbol,
                expected=expected,
                received=sequence,
                gap_size=gap_size
            )
            self.gaps_detected += 1

            # Actualizar de todas formas
            self.last_sequences[symbol] = sequence
            return False

        self.last_sequences[symbol] = sequence
        return True