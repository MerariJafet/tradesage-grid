#!/bin/bash
# Script para cambiar entre entornos

BACKEND_DIR="/Users/merari/Desktop/bot de scalping/backend"

if [ "$1" == "production" ]; then
    echo "⚠️  CAMBIANDO A PRODUCCIÓN (DINERO REAL)"
    echo "⚠️  ¿Estás seguro? (escribe 'SI' para confirmar)"
    read confirmation
    
    if [ "$confirmation" == "SI" ]; then
        cp "$BACKEND_DIR/.env.production" "$BACKEND_DIR/.env"
        echo "✅ Ahora usando .env.production"
        echo "💰 MODO: PRODUCCIÓN - DINERO REAL"
    else
        echo "❌ Operación cancelada"
    fi
    
elif [ "$1" == "testnet" ]; then
    echo "🔄 Cambiando a testnet (desarrollo)..."
    
    # Backup del .env actual si existe
    if [ -f "$BACKEND_DIR/.env" ]; then
        cp "$BACKEND_DIR/.env" "$BACKEND_DIR/.env.backup"
    fi
    
    # Crear .env de testnet
    cat > "$BACKEND_DIR/.env" << 'ENVEOF'
MODE=paper
BINANCE_API_KEY=PENDIENTE_KEY_DE_TESTNET
BINANCE_API_SECRET=PENDIENTE_SECRET_DE_TESTNET
BINANCE_TESTNET=true
POSTGRES_USER=tradesage
POSTGRES_PASSWORD=test_password_123
POSTGRES_DB=tradesage_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
REDIS_HOST=redis
REDIS_PORT=6379
BACKEND_PORT=8000
FRONTEND_PORT=3000
LOG_LEVEL=INFO
MAX_RISK_PER_TRADE_PCT=0.5
MAX_DAILY_LOSS_PCT=2.0
MAX_CONSECUTIVE_LOSSES=3
ENABLE_TELEMETRY=true
TELEMETRY_LOG_INTERVAL=60
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENVEOF
    
    echo "✅ Ahora usando .env (testnet)"
    echo "🎮 MODO: TESTNET - DINERO VIRTUAL"
    
else
    echo "❌ Uso: ./switch_env.sh [testnet|production]"
    echo ""
    echo "Ejemplo:"
    echo "  ./switch_env.sh testnet     # Para desarrollo (seguro)"
    echo "  ./switch_env.sh production  # Para producción (dinero real)"
fi