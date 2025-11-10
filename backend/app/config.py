from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Modo de operación
    MODE: Literal["paper", "live"] = "paper"
    
    # Binance
    BINANCE_API_KEY: str
    BINANCE_API_SECRET: str
    BINANCE_TESTNET: bool = True
    
    # Database
    POSTGRES_USER: str = "tradesage"
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str = "tradesage_db"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    
    @property
    def DATABASE_URL(self) -> str:
        # Use SQLite for testing, PostgreSQL for production
        if self.MODE == "test":
            return "sqlite+aiosqlite:///./tradesage_test.db"
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str = "redis-1"
    REDIS_PORT: int = 6379
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"
    
    # App
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 3000
    LOG_LEVEL: str = "INFO"
    
    # Risk Management
    MAX_RISK_PER_TRADE_PCT: float = 0.5  # 0.5% del capital
    MAX_DAILY_LOSS_PCT: float = 2.0
    MAX_CONSECUTIVE_LOSSES: int = 3
    
    # Telemetry
    ENABLE_TELEMETRY: bool = True
    TELEMETRY_LOG_INTERVAL: int = 60

    # Telegram (opcional)
    TELEGRAM_ENABLED: bool = False
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()