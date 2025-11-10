from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

# DTO unificado para ticks/trades
class Tick(BaseModel):
    timestamp: datetime
    exchange: str = "binance"
    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: Optional[int] = None
    latency_ms: Optional[float] = None  # Calculado: ahora - exchange_time

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-15T10:30:00Z",
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "price": 62000.50,
                "quantity": 0.05,
                "is_buyer_maker": True,
                "trade_id": 123456789,
                "latency_ms": 45.2
            }
        }

# DTO para barras OHLCV
class Bar(BaseModel):
    timestamp: datetime
    exchange: str = "binance"
    symbol: str
    timeframe: str  # 1m, 5m, 15m, 1h
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool  # True si la vela está cerrada

# DTO para snapshots de order book
class OrderBookSnapshot(BaseModel):
    timestamp: datetime
    exchange: str = "binance"
    symbol: str
    bids: list[tuple[float, float]]  # [(price, quantity), ...]
    asks: list[tuple[float, float]]
    last_update_id: int

    def calculate_obi(self, depth: int = 5) -> float:
        """Calcular Order Book Imbalance en top N niveles"""
        bid_volume = sum(qty for _, qty in self.bids[:depth])
        ask_volume = sum(qty for _, qty in self.asks[:depth])
        total = bid_volume + ask_volume
        return bid_volume / total if total > 0 else 0.5

# Mensajes raw de Binance
class BinanceAggTradeMessage(BaseModel):
    e: str = Field(alias="e")  # Event type
    E: int = Field(alias="E")  # Event time
    s: str = Field(alias="s")  # Symbol
    a: int = Field(alias="a")  # Aggregate trade ID
    p: str = Field(alias="p")  # Price
    q: str = Field(alias="q")  # Quantity
    f: int = Field(alias="f")  # First trade ID
    l: int = Field(alias="l")  # Last trade ID
    T: int = Field(alias="T")  # Trade time
    m: bool = Field(alias="m")  # Is buyer maker

class BinanceKlineMessage(BaseModel):
    e: str  # kline
    E: int  # Event time
    s: str  # Symbol
    k: dict  # Kline data

class BinanceDepthMessage(BaseModel):
    e: str  # depthUpdate
    E: int  # Event time
    s: str  # Symbol
    U: int  # First update ID
    u: int  # Final update ID
    b: list[list[str]]  # Bids
    a: list[list[str]]  # Asks