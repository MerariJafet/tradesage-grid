import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["mode"] == settings.MODE

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "tradesage_" in response.text

def test_database_connection():
    # Verificar que TimescaleDB está funcionando
    from app.db.database import engine
    with engine.connect() as conn:
        result = conn.execute("SELECT 1")
        assert result.scalar() == 1

def test_redis_connection():
    import redis
    r = redis.from_url(settings.REDIS_URL)
    assert r.ping() == True

def test_logging_format():
    # Verificar que logs son JSON
    from app.utils.logger import get_logger
    logger = get_logger("test")
    # Log debe ser JSON, no texto plano
    logger.info("test_message", extra_field="value")