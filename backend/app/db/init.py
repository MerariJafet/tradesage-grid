from sqlalchemy import create_engine
from app.config import settings
from app.db.models import Base

def init_db():
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized")