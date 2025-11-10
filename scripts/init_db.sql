-- Habilitar TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tabla de usuarios (single-user por ahora)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla de API keys encriptadas (preparado para futuro)
CREATE TABLE api_keys_encrypted (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    exchange VARCHAR(20) NOT NULL,
    api_key_encrypted BYTEA NOT NULL,
    api_secret_encrypted BYTEA NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_rotated TIMESTAMPTZ DEFAULT NOW()
);

-- Hipertabla para datos de mercado (ticks)
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    is_buyer_maker BOOLEAN,
    trade_id BIGINT,
    PRIMARY KEY (time, exchange, symbol)
);

SELECT create_hypertable('market_data', 'time');

-- Índices para búsquedas rápidas
CREATE INDEX idx_market_data_symbol ON market_data (symbol, time DESC);

-- Hipertabla para barras (OHLCV)
CREATE TABLE bars (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,  -- 1m, 5m, 15m, 1h
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    PRIMARY KEY (time, exchange, symbol, timeframe)
);

SELECT create_hypertable('bars', 'time');

-- Tabla de trades ejecutados
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL,  -- buy, sell
    order_type VARCHAR(10) NOT NULL,  -- market, limit
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    filled_price NUMERIC(20, 8),
    commission NUMERIC(20, 8),
    slippage NUMERIC(10, 4),
    pnl NUMERIC(20, 8),
    status VARCHAR(20) NOT NULL,  -- pending, filled, rejected
    mode VARCHAR(10) NOT NULL,  -- paper, live
    created_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ
);

CREATE INDEX idx_trades_user_created ON trades(user_id, created_at DESC);
CREATE INDEX idx_trades_symbol_strategy ON trades(symbol, strategy_name, created_at DESC);

-- Tabla de audit log (inmutable)
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id INTEGER,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_created ON audit_log(created_at DESC);

-- Tabla para order book snapshots (nueva)
CREATE TABLE order_book_snapshots (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    best_bid NUMERIC(20, 8) NOT NULL,
    best_ask NUMERIC(20, 8) NOT NULL,
    bid_volume_5 NUMERIC(20, 8),  -- Volumen top 5 niveles
    ask_volume_5 NUMERIC(20, 8),
    obi_5 NUMERIC(5, 4),  -- Order Book Imbalance
    obi_10 NUMERIC(5, 4),
    spread_pct NUMERIC(10, 6),
    last_update_id BIGINT,
    PRIMARY KEY (time, exchange, symbol)
);

SELECT create_hypertable('order_book_snapshots', 'time');
CREATE INDEX idx_ob_symbol_time ON order_book_snapshots (symbol, time DESC);

-- Retention policies
SELECT add_retention_policy('order_book_snapshots', INTERVAL '3 days');  -- OB muy voluminoso

-- Continuous aggregates para barras mayores
CREATE MATERIALIZED VIEW bars_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS time,
    exchange,
    symbol,
    '5m' AS timeframe,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM bars
WHERE timeframe = '1m'
GROUP BY time_bucket('5 minutes', time), exchange, symbol;

SELECT add_continuous_aggregate_policy('bars_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');