# services/api-service/tests/test_api_endpoints.py
# Unit tests for API service endpoints

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, MediaItem, KeywordHit, Transcript # Import necessary models

# Test database
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override database dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

# Setup/Teardown
@pytest.fixture(autouse=True)
def setup_database():
    """Create tables before each test"""
    Base.metadata.create_all(bind=engine)
    
    # Add some dummy data for tests
    db = TestingSessionLocal()
    try:
        media_item = MediaItem(
            video_id="test_video_1",
            filename="test_video_1.mp4",
            status="completed"
        )
        db.add(media_item)
        db.commit()
        db.refresh(media_item)

        keyword_hit = KeywordHit(
            media_item_id=media_item.id,
            keyword="RSI",
            category="technical_indicator",
            start_time=10.0,
            end_time=12.0,
            confidence=0.95
        )
        db.add(keyword_hit)
        db.commit()
        db.refresh(keyword_hit)

        transcript = Transcript(
            media_item_id=media_item.id,
            segment_index=0,
            start_time=0.0,
            end_time=5.0,
            text="This is a test transcript segment about RSI."
        )
        db.add(transcript)
        db.commit()

    finally:
        db.close()

    yield
    Base.metadata.drop_all(bind=engine)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Health endpoint should return 200 with status"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data


class TestMediaItemsEndpoint:
    """Test media items endpoints"""
    
    def test_list_media_items_empty(self):
        """Should return empty list when no media items"""
        # Drop all tables first to ensure it's empty
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        response = client.get("/media_items")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 0
    
    def test_get_media_item_not_found(self):
        """Should return 404 for non-existent media item"""
        response = client.get("/media_items/999")
        assert response.status_code == 404


class TestIngestEndpoint:
    """Test video ingest endpoint"""
    
    def test_ingest_invalid_path(self):
        """Should reject invalid video path"""
        response = client.post(
            "/ingest",
            json={
                "video_path": "/invalid/path.mp4",
                "filename": "test.mp4"
            }
        )
        # May return 400 or 500 depending on validation
        assert response.status_code in [400, 500]
    
    def test_ingest_missing_filename(self):
        """Should reject missing filename"""
        response = client.post(
            "/ingest",
            json={
                "video_path": "/data/videos/test.mp4"
            }
        )
        assert response.status_code == 422  # Validation error


class TestKeywordsEndpoint:
    """Test keywords endpoint"""
    
    def test_keywords_empty(self):
        """Should return empty list when no keywords"""
        # Drop all tables first to ensure it's empty
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        response = client.get("/keywords")
        assert response.status_code == 200
        
        data = response.json()
        assert "keywords" in data
        assert len(data["keywords"]) == 0


class TestClipsEndpoint:
    """Test clips endpoint"""
    
    def test_clips_filter_by_keyword(self):
        """Should accept keyword filter parameter"""
        response = client.get("/clips?keyword=RSI")
        assert response.status_code == 200
        
        data = response.json()
        assert "clips" in data


class TestLlamaEndpoints:
    """Test Llama dataset API endpoints"""
    
    def test_llama_examples_no_keyword(self):
        """Should return examples without keyword filter"""
        response = client.get("/llama/examples")
        assert response.status_code == 200
        
        data = response.json()
        assert "examples" in data
        assert "total" in data
        assert len(data["examples"]) > 0 # Should have the dummy data
    
    def test_llama_examples_with_keyword(self):
        """Should filter by keyword"""
        response = client.get("/llama/examples?keyword=RSI&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["examples"]) > 0
        assert data["examples"][0]["keyword"] == "RSI"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])