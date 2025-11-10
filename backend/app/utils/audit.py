from sqlalchemy.orm import Session
from app.db.models import AuditLog
from typing import Optional, Dict
import json
from app.utils.logger import get_logger

class AuditLogger:
    def __init__(self, db: Session):
        self.db = db
    
    def log(
        self,
        user_id: Optional[int],
        action: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None
    ):
        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            ip_address=ip_address
        )
        self.db.add(audit_entry)
        self.db.commit()
        
        # También loguear a structlog
        logger = get_logger("audit")
        logger.info(
            "audit_event",
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id
        )