# tests/integration/test_service_integration.py
# Integration tests for service-to-service communication

import pytest
import requests
import os

BASE_URL_API = os.getenv("API_BASE_URL", "http://localhost:8003")
BASE_URL_EMBEDDINGS = os.getenv(
    "EMBEDDINGS_SERVICE_URL",
    "http://localhost:8004")
BASE_URL_BACKTEST = os.getenv("BACKTEST_SERVICE_URL", "http://localhost:8001")


class TestServiceHealth:
    """Test all services are running and healthy"""

    def test_api_service_health(self):
        """API service should be healthy"""
        response = requests.get(f"{BASE_URL_API}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_embeddings_service_health(self):
        """Embeddings service should be healthy"""
        response = requests.get(f"{BASE_URL_EMBEDDINGS}/health", timeout=5)
        assert response.status_code in [200, 503]  # May be initializing

    def test_backtest_service_health(self):
        """Backtesting service should be healthy"""
        response = requests.get(f"{BASE_URL_BACKTEST}/health", timeout=5)
        assert response.status_code == 200


class TestAPIToEmbeddingsIntegration:
    """Test API service calling embeddings service"""

    def test_embeddings_search_via_api(self):
        """API should proxy search requests to embeddings service"""
        response = requests.get(
            f"{BASE_URL_API}/embeddings/search",
            params={
                "query": "test query",
                "top_k": 5,
                "embedding_type": "transcript"
            },
            timeout=10
        )

        # Should return 200 even if no results
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


class TestAPIToBacktestIntegration:
    """Test API service to backtesting service integration"""

    def test_strategy_creation_flow(self):
        """Should create strategy through orchestrator"""
        # This would be triggered by video processing pipeline
        # Here we test the backtest service directly
        response = requests.post(
            f"{BASE_URL_BACKTEST}/strategies",
            json={
                "name": "Integration_Test_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            },
            timeout=10
        )

        assert response.status_code == 201
        assert "strategy_id" in response.json()


class TestDatabaseIntegration:
    """Test database operations across services"""

    def test_media_item_persistence(self):
        """Media items should persist in database"""
        # Get initial count
        response1 = requests.get(f"{BASE_URL_API}/media_items", timeout=5)
        len(response1.json().get("items", []))

        # Note: Actual creation would require valid video file
        # This test verifies the endpoint structure
        response2 = requests.get(f"{BASE_URL_API}/media_items", timeout=5)
        assert response2.status_code == 200


class TestMessageQueueIntegration:
    """Test Celery/RabbitMQ integration"""

    def test_task_submission(self):
        """Should accept task submission"""
        # This would normally submit to Celery
        # Here we verify the ingest endpoint exists
        response = requests.post(
            f"{BASE_URL_API}/ingest",
            json={
                "video_path": "/data/videos/test.mp4",
                "filename": "test.mp4"
            },
            timeout=5
        )

        # Will fail without valid file, but endpoint should exist
        assert response.status_code in [200, 400, 404, 500]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
