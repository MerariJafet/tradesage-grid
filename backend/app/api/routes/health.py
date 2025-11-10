from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.config import settings
import redis
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    checks = {
        "status": "healthy",
        "mode": settings.MODE,
        "database": "unknown",
        "redis": "unknown"
    }
    
    # Check DB
    try:
        db.execute("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
        checks["status"] = "degraded"
    
    # Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
        checks["status"] = "degraded"
    
    return checks

@router.get("/health/ready")
async def readiness_check():
    # Para Kubernetes readiness probes
    return {"ready": True}

@router.get("/health/live")
async def liveness_check():
    # Para Kubernetes liveness probes
    return {"alive": True}