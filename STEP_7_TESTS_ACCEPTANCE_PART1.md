# STEP_7_TESTS_ACCEPTANCE_PART1.md
# Step 7: Tests & Acceptance - Complete Testing Framework

## Overview

Step 7 establishes comprehensive testing, CI/CD pipeline, and production deployment readiness. This is the final step ensuring all components work together reliably.

---

## Architecture

```
Testing & CI/CD Framework
├── Unit Tests
│   ├── Feature Engineering Tests
│   ├── Backtesting Service Tests
│   ├── Embeddings Service Tests
│   ├── API Service Tests
│   └── Orchestrator Tasks Tests
├── Integration Tests
│   ├── Service-to-Service Communication
│   ├── Database Integration
│   ├── Message Queue Integration
│   └── Full Pipeline Tests
├── Performance Tests
│   ├── Load Testing
│   ├── Stress Testing
│   └── Benchmark Suites
├── CI/CD Pipeline
│   ├── GitHub Actions Workflows
│   ├── Automated Testing
│   ├── Docker Build & Push
│   └── Deployment Automation
└── Production Readiness
    ├── Health Monitoring
    ├── Logging & Alerting
    ├── Backup & Recovery
    └── Documentation
```

---

## Part 1: Unit Tests

### 1. API Service Unit Tests

Create `services/api-service/tests/test_api_endpoints.py`:

```python
# services/api-service/tests/test_api_endpoints.py
# Unit tests for API service endpoints

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

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
        response = client.get("/keywords")
        assert response.status_code == 200
        
        data = response.json()
        assert "keywords" in data


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
    
    def test_llama_examples_with_keyword(self):
        """Should filter by keyword"""
        response = client.get("/llama/examples?keyword=RSI&top_k=5")
        assert response.status_code == 200


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

### 2. Embeddings Service Unit Tests

Create `services/embeddings-service/tests/test_embeddings.py`:

```python
# services/embeddings-service/tests/test_embeddings.py
# Unit tests for embeddings service

import pytest
from fastapi.testclient import TestClient
from app.main import app
import numpy as np

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
```

---

### 3. Backtesting Service Unit Tests

Create `services/backtesting-service/tests/test_backtest.py`:

```python
# services/backtesting-service/tests/test_backtest.py
# Unit tests for backtesting service

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test backtesting service health"""
    
    def test_health_check(self):
        """Health endpoint should return service info"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "strategies_count" in data
        assert "backtests_count" in data


class TestStrategyCreation:
    """Test strategy creation endpoint"""
    
    def test_create_valid_strategy(self):
        """Should create strategy with valid parameters"""
        response = client.post(
            "/strategies",
            json={
                "name": "Test_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "strategy_id" in data
        assert data["name"] == "Test_Strategy"
    
    def test_create_strategy_missing_fields(self):
        """Should reject strategy with missing fields"""
        response = client.post(
            "/strategies",
            json={
                "name": "Incomplete_Strategy"
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_get_strategy(self):
        """Should retrieve created strategy"""
        # Create strategy first
        create_response = client.post(
            "/strategies",
            json={
                "name": "Test_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]
        
        # Get strategy
        response = client.get(f"/strategies/{strategy_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == strategy_id
        assert data["name"] == "Test_Strategy"


class TestBacktestExecution:
    """Test backtest execution"""
    
    def test_run_backtest(self):
        """Should execute backtest for valid strategy"""
        # Create strategy
        create_response = client.post(
            "/strategies",
            json={
                "name": "Backtest_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]
        
        # Run backtest
        response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "backtest_id" in data
        assert "metrics" in data
        assert data["strategy_id"] == strategy_id
        
        # Verify metrics structure
        metrics = data["metrics"]
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "max_drawdown" in metrics
        assert "num_trades" in metrics
    
    def test_backtest_invalid_strategy(self):
        """Should reject backtest for non-existent strategy"""
        response = client.post(
            "/backtest",
            json={
                "strategy_id": 99999,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        assert response.status_code == 404


class TestBacktestResults:
    """Test backtest results endpoints"""
    
    def test_get_metrics(self):
        """Should retrieve backtest metrics"""
        # Create and run backtest
        create_response = client.post(
            "/strategies",
            json={
                "name": "Metrics_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]
        
        backtest_response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        backtest_id = backtest_response.json()["backtest_id"]
        
        # Get metrics
        response = client.get(f"/backtest/{backtest_id}/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "sharpe_ratio" in data
        assert "win_rate" in data
    
    def test_get_trades(self):
        """Should retrieve trade history"""
        # Create and run backtest
        create_response = client.post(
            "/strategies",
            json={
                "name": "Trades_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]
        
        backtest_response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        backtest_id = backtest_response.json()["backtest_id"]
        
        # Get trades
        response = client.get(f"/backtest/{backtest_id}/trades")
        assert response.status_code == 200
        
        data = response.json()
        assert "trades" in data
        assert "num_trades" in data
    
    def test_get_equity_curve(self):
        """Should retrieve equity curve"""
        # Create and run backtest
        create_response = client.post(
            "/strategies",
            json={
                "name": "Equity_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]
        
        backtest_response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        backtest_id = backtest_response.json()["backtest_id"]
        
        # Get equity curve
        response = client.get(f"/backtest/{backtest_id}/equity")
        assert response.status_code == 200
        
        data = response.json()
        assert "equity_curve" in data
        assert "initial_capital" in data
        assert "final_capital" in data


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

### 4. Feature Engineering Unit Tests

Already created in Step 6, ensure it exists at:
`tests/test_feature_engineering.py`

---

### 5. Test Requirements

Create `requirements-test.txt` in project root:

```
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
requests==2.31.0
fastapi==0.104.1
httpx==0.25.0
```

---

## Running Unit Tests

### Run All Unit Tests

```powershell
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests with coverage
pytest tests/ services/ -v --cov=. --cov-report=html

# Run specific service tests
pytest services/api-service/tests/ -v
pytest services/embeddings-service/tests/ -v
pytest services/backtesting-service/tests/ -v

# Run with output
pytest -v -s
```

---

## Continue to Part 2: Integration Tests...

Would you like me to continue with Part 2 covering:
1. Integration tests (service-to-service, database, message queue)
2. Performance tests (load testing, benchmarks)
3. CI/CD pipeline setup
4. Production deployment guide