#!/usr/bin/env python
"""
Database initialization script
Creates all tables defined in models
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/app')

from app.models import Base
from app.database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """
    Initialize database with all tables
    
    Creates tables from SQLAlchemy models if they don't exist
    """
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created successfully")
        
        # List created tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"✓ Tables created: {', '.join(tables)}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        return False

def seed_data():
    """
    Insert initial/test data (optional)
    """
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        
        # Add seed data if needed
        # Example: Create default categories, etc.
        
        db.commit()
        logger.info("✓ Seed data inserted successfully")
    
    except Exception as e:
        logger.error(f"❌ Seed data failed: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    success = init_database()
    
    # Optionally seed data
    # seed_data()
    
    sys.exit(0 if success else 1)
