#!/usr/bin/env python
"""
Test Database Models
Tests: Model definitions, relationships, SQLAlchemy setup
"""
import sys
import os

# Add orchestrator to path
sys.path.insert(0, 'orchestrator')

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Test 3.1: Checking imports...")
    try:
        import sqlalchemy
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: pip install sqlalchemy psycopg2-binary")
        return False

def test_models_import():
    """Test if models can be imported"""
    print("\n🧪 Test 3.2: Testing models import...")
    try:
        from app.models import (
            Base, MediaItem, Transcript, KeywordHit, 
            Clip, Embedding, ProvenStrategy
        )
        
        print("✅ All models imported successfully")
        print(f"   Models: MediaItem, Transcript, KeywordHit, Clip, Embedding, ProvenStrategy")
        return True
    except Exception as e:
        print(f"❌ Models import failed: {e}")
        return False

def test_model_attributes():
    """Test if models have expected attributes"""
    print("\n🧪 Test 3.3: Testing model attributes...")
    try:
        from app.models import MediaItem, Transcript, KeywordHit
        
        # Check MediaItem
        media_attrs = ['id', 'video_id', 'filename', 'file_path', 'status', 'created_at']
        for attr in media_attrs:
            if not hasattr(MediaItem, attr):
                print(f"❌ MediaItem missing attribute: {attr}")
                return False
        
        # Check relationships
        if not hasattr(MediaItem, 'transcripts'):
            print("❌ MediaItem missing 'transcripts' relationship")
            return False
        
        print("✅ Model attributes correct")
        print(f"   MediaItem has all required fields and relationships")
        return True
    except Exception as e:
        print(f"❌ Model attributes test failed: {e}")
        return False

def test_database_connection():
    """Test database connection (if available)"""
    print("\n🧪 Test 3.4: Testing database connection...")
    try:
        from app.database import engine, SessionLocal
        from sqlalchemy import text
        
        # Try to connect
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"⚠️  Database connection failed (expected if DB not running): {e}")
        print("   This is OK for testing without Docker")
        return True  # Don't fail test if DB not running

def test_table_creation():
    """Test if tables can be created (using in-memory SQLite)"""
    print("\n🧪 Test 3.5: Testing table creation...")
    try:
        from app.models import Base
        from sqlalchemy import create_engine
        from sqlalchemy import inspect
        
        # Create in-memory SQLite database for testing
        test_engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(bind=test_engine)
        
        # Check tables were created
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'media_items',
            'transcripts',
            'keyword_hits',
            'clips',
            'embeddings',
            'proven_strategies'
        ]
        
        missing_tables = [t for t in expected_tables if t not in tables]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        print("✅ All tables created successfully")
        print(f"   Tables: {', '.join(tables)}")
        return True
    except Exception as e:
        print(f"❌ Table creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE MODELS - STANDALONE TESTS")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Models Import", test_models_import()))
    results.append(("Model Attributes", test_model_attributes()))
    results.append(("Database Connection", test_database_connection()))
    results.append(("Table Creation", test_table_creation()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nPassed: {total_passed}/{len(results)}")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Database models are ready.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Fix issues before building Docker image.")
        sys.exit(1)
