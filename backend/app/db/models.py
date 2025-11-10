from sqlalchemy import Column, Integer, String, Numeric, Boolean, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, INET

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ApiKeyEncrypted(Base):
    __tablename__ = "api_keys_encrypted"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exchange = Column(String(20), nullable=False)
    api_key_encrypted = Column(LargeBinary, nullable=False)
    api_secret_encrypted = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_rotated = Column(DateTime(timezone=True), server_default=func.now())

class MarketData(Base):
    __tablename__ = "market_data"
    time = Column(DateTime(timezone=True), primary_key=True)
    exchange = Column(String(20), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    is_buyer_maker = Column(Boolean)
    trade_id = Column(Integer)

class Bar(Base):
    __tablename__ = "bars"
    time = Column(DateTime(timezone=True), primary_key=True)
    exchange = Column(String(20), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    timeframe = Column(String(5), primary_key=True)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(20, 8), nullable=False)

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    strategy_name = Column(String(50), nullable=False)
    side = Column(String(4), nullable=False)
    order_type = Column(String(10), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    filled_price = Column(Numeric(20, 8))
    commission = Column(Numeric(20, 8))
    slippage = Column(Numeric(10, 4))
    pnl = Column(Numeric(20, 8))
    status = Column(String(20), nullable=False)
    mode = Column(String(10), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    filled_at = Column(DateTime(timezone=True))

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    action = Column(String(50), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(Integer)
    details = Column(JSONB)
    ip_address = Column(INET)
    created_at = Column(DateTime(timezone=True), server_default=func.now())