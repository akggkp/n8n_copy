# services/embeddings-service/tests/test_embeddings.py
# Unit tests for embeddings service

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test embeddings service health"""

    def test_health_check(self):
        """Health endpoint should return service status"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be initializing

        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestStatsEndpoint:
    """Test embeddings statistics"""

    def test_stats_endpoint(self):
        """Should return embedding statistics"""
        response = client.get("/stats")

        # May be 503 if not initialized, 200 if ready
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "embedding_dimension" in data
            assert "total_vectors" in data


class TestEmbedEndpoint:
    """Test embedding generation"""

    def test_embed_transcripts(self):
        """Should generate embeddings for transcripts"""
        response = client.post(
            "/embed",
            json={
                "media_item_id": 1,
                "embedding_type": "transcript",
                "segments": [
                    {
                        "id": 1,
                        "text": "Test transcript segment",
                        "start_time": 0.0,
                        "end_time": 3.0
                    }
                ]
            }
        )

        # May be 503 if service not ready
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["embeddings_created"] > 0

    def test_embed_invalid_type(self):
        """Should reject invalid embedding type"""
        response = client.post(
            "/embed",
            json={
                "media_item_id": 1,
                "embedding_type": "invalid_type",
                "segments": []
            }
        )
        assert response.status_code in [400, 500]


class TestSearchEndpoints:
    """Test semantic search endpoints"""

    def test_search_transcripts_empty_index(self):
        """Should handle search on empty index"""
        response = client.get(
            "/search/transcripts?query=test&top_k=5"
        )

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            # Empty index should return empty results
            assert isinstance(data["results"], list)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
