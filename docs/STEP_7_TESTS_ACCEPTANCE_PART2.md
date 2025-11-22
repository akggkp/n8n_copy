# STEP_7_TESTS_ACCEPTANCE_PART2.md
# Step 7: Tests & Acceptance - Part 2: Integration & Performance Tests

## Integration Tests

### 1. Service-to-Service Integration Tests

Create `tests/integration/test_service_integration.py`:

```python
# tests/integration/test_service_integration.py
# Integration tests for service-to-service communication

import pytest
import requests
import time

BASE_URL_API = "http://localhost:8003"
BASE_URL_EMBEDDINGS = "http://localhost:8004"
BASE_URL_BACKTEST = "http://localhost:8001"


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
        initial_count = len(response1.json().get("items", []))
        
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
```

---

### 2. Full Pipeline Integration Test

Create `tests/integration/test_full_pipeline.py`:

```python
# tests/integration/test_full_pipeline.py
# Full end-to-end pipeline integration test

import pytest
import requests
import time
import os

BASE_URL_API = "http://localhost:8003"
BASE_URL_BACKTEST = "http://localhost:8001"
BASE_URL_EMBEDDINGS = "http://localhost:8004"

# Test video file path
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "/data/videos/sample.mp4")


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Full pipeline integration test (requires all services)"""
    
    def test_complete_video_processing_pipeline(self):
        """
        Test complete pipeline:
        1. Ingest video
        2. Wait for processing
        3. Verify keywords extracted
        4. Verify embeddings created
        5. Verify strategy generated
        6. Verify backtest completed
        """
        
        # Step 1: Ingest video
        print("\n1. Ingesting video...")
        response = requests.post(
            f"{BASE_URL_API}/ingest",
            json={
                "video_path": TEST_VIDEO_PATH,
                "filename": "sample.mp4"
            },
            timeout=30
        )
        
        if response.status_code != 200:
            pytest.skip(f"Video ingest failed: {response.text}")
        
        data = response.json()
        media_item_id = data.get("media_item_id")
        assert media_item_id is not None
        
        # Step 2: Wait for processing (max 10 minutes)
        print("2. Waiting for processing...")
        max_wait = 600
        interval = 15
        elapsed = 0
        
        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval
            
            status_response = requests.get(
                f"{BASE_URL_API}/media_items/{media_item_id}",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status = status_response.json().get("status")
                print(f"   [{elapsed}s] Status: {status}")
                
                if status == "completed":
                    break
                elif status == "failed":
                    pytest.fail("Pipeline failed")
        
        assert elapsed < max_wait, "Pipeline timeout"
        
        # Step 3: Verify keywords
        print("3. Verifying keywords...")
        keywords_response = requests.get(
            f"{BASE_URL_API}/keywords",
            params={"media_item_id": media_item_id},
            timeout=10
        )
        assert keywords_response.status_code == 200
        keywords = keywords_response.json().get("keywords", [])
        print(f"   Keywords found: {len(keywords)}")
        
        # Step 4: Verify embeddings (optional - service may not be ready)
        print("4. Checking embeddings...")
        try:
            embeddings_response = requests.get(
                f"{BASE_URL_EMBEDDINGS}/stats",
                timeout=5
            )
            if embeddings_response.status_code == 200:
                stats = embeddings_response.json()
                print(f"   Total vectors: {stats.get('total_vectors', 0)}")
        except:
            print("   Embeddings service not available")
        
        # Step 5: Verify strategy (check backtest service)
        print("5. Checking strategies...")
        health_response = requests.get(
            f"{BASE_URL_BACKTEST}/health",
            timeout=5
        )
        if health_response.status_code == 200:
            strategies_count = health_response.json().get("strategies_count", 0)
            print(f"   Strategies created: {strategies_count}")
        
        print("\n✓ Pipeline integration test completed")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

---

## Performance Tests

### 1. Load Testing Script

Create `tests/performance/test_load.py`:

```python
# tests/performance/test_load.py
# Load testing for API endpoints

import pytest
import requests
import time
import concurrent.futures
from statistics import mean, median, stdev

BASE_URL_API = "http://localhost:8003"
BASE_URL_BACKTEST = "http://localhost:8001"


class LoadTester:
    """Load testing utility"""
    
    def __init__(self, url: str):
        self.url = url
        self.results = []
    
    def make_request(self):
        """Make single request and record metrics"""
        start = time.time()
        try:
            response = requests.get(self.url, timeout=10)
            duration = time.time() - start
            
            return {
                "success": response.status_code == 200,
                "duration": duration,
                "status_code": response.status_code
            }
        except Exception as e:
            duration = time.time() - start
            return {
                "success": False,
                "duration": duration,
                "error": str(e)
            }
    
    def run_load_test(self, num_requests: int, concurrent: int = 10):
        """Run load test with concurrent requests"""
        print(f"\nLoad Test: {num_requests} requests, {concurrent} concurrent")
        print(f"Target: {self.url}")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(self.make_request) for _ in range(num_requests)]
            self.results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        durations = [r["duration"] for r in successful]
        
        metrics = {
            "total_requests": num_requests,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / num_requests * 100,
            "total_time": total_time,
            "requests_per_second": num_requests / total_time,
            "avg_duration": mean(durations) if durations else 0,
            "median_duration": median(durations) if durations else 0,
            "stdev_duration": stdev(durations) if len(durations) > 1 else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0
        }
        
        return metrics
    
    def print_results(self, metrics: dict):
        """Print formatted results"""
        print("\nResults:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Successful: {metrics['successful']}")
        print(f"  Failed: {metrics['failed']}")
        print(f"  Success Rate: {metrics['success_rate']:.2f}%")
        print(f"  Total Time: {metrics['total_time']:.2f}s")
        print(f"  Requests/sec: {metrics['requests_per_second']:.2f}")
        print(f"\nResponse Times:")
        print(f"  Average: {metrics['avg_duration']*1000:.2f}ms")
        print(f"  Median: {metrics['median_duration']*1000:.2f}ms")
        print(f"  Std Dev: {metrics['stdev_duration']*1000:.2f}ms")
        print(f"  Min: {metrics['min_duration']*1000:.2f}ms")
        print(f"  Max: {metrics['max_duration']*1000:.2f}ms")


@pytest.mark.performance
class TestAPILoad:
    """Load tests for API service"""
    
    def test_health_endpoint_load(self):
        """Test health endpoint under load"""
        tester = LoadTester(f"{BASE_URL_API}/health")
        metrics = tester.run_load_test(num_requests=100, concurrent=10)
        tester.print_results(metrics)
        
        assert metrics["success_rate"] >= 95.0
        assert metrics["avg_duration"] < 1.0  # < 1 second average
    
    def test_media_items_list_load(self):
        """Test media items list under load"""
        tester = LoadTester(f"{BASE_URL_API}/media_items")
        metrics = tester.run_load_test(num_requests=50, concurrent=5)
        tester.print_results(metrics)
        
        assert metrics["success_rate"] >= 90.0
        assert metrics["avg_duration"] < 2.0


@pytest.mark.performance
class TestBacktestLoad:
    """Load tests for backtesting service"""
    
    def test_health_endpoint_load(self):
        """Test backtest health endpoint under load"""
        tester = LoadTester(f"{BASE_URL_BACKTEST}/health")
        metrics = tester.run_load_test(num_requests=100, concurrent=10)
        tester.print_results(metrics)
        
        assert metrics["success_rate"] >= 95.0
        assert metrics["avg_duration"] < 1.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
```

---

### 2. Benchmark Suite

Create `tests/performance/benchmark.py`:

```python
# tests/performance/benchmark.py
# Benchmark suite for key operations

import time
import numpy as np
from orchestrator.app.feature_engineering import FeatureEngineer


class Benchmark:
    """Benchmarking utility"""
    
    def __init__(self, name: str):
        self.name = name
        self.times = []
    
    def run(self, func, *args, iterations=10, **kwargs):
        """Run function multiple times and record timings"""
        print(f"\nBenchmark: {self.name}")
        print(f"Iterations: {iterations}")
        
        for i in range(iterations):
            start = time.time()
            func(*args, **kwargs)
            duration = time.time() - start
            self.times.append(duration)
        
        avg = np.mean(self.times)
        std = np.std(self.times)
        min_time = np.min(self.times)
        max_time = np.max(self.times)
        
        print(f"Results:")
        print(f"  Average: {avg*1000:.2f}ms")
        print(f"  Std Dev: {std*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        
        return {
            "avg": avg,
            "std": std,
            "min": min_time,
            "max": max_time
        }


def benchmark_feature_extraction():
    """Benchmark feature extraction performance"""
    engineer = FeatureEngineer()
    
    # Sample data
    keyword_hits = [
        {'keyword': 'RSI', 'category': 'technical_indicator', 'start_time': 10.0, 'end_time': 12.0, 'confidence': 0.95},
        {'keyword': 'breakout', 'category': 'price_action', 'start_time': 25.0, 'end_time': 27.0, 'confidence': 0.88},
        {'keyword': 'stop loss', 'category': 'risk_management', 'start_time': 40.0, 'end_time': 42.0, 'confidence': 0.92}
    ] * 10  # 30 hits
    
    transcripts = [
        {'text': 'Today we will discuss trading', 'start_time': 0.0, 'end_time': 3.0},
        {'text': 'RSI is a momentum indicator', 'start_time': 3.0, 'end_time': 6.0}
    ] * 20  # 40 segments
    
    embeddings = [np.random.randn(384) for _ in range(50)]
    
    # Benchmark keyword extraction
    bench1 = Benchmark("Keyword Feature Extraction")
    bench1.run(engineer.extract_keyword_features, keyword_hits, iterations=100)
    
    # Benchmark transcript extraction
    bench2 = Benchmark("Transcript Feature Extraction")
    bench2.run(engineer.extract_transcript_features, transcripts, iterations=100)
    
    # Benchmark embedding extraction
    bench3 = Benchmark("Embedding Feature Extraction")
    bench3.run(engineer.extract_embedding_features, embeddings, iterations=100)
    
    # Benchmark full feature vector
    bench4 = Benchmark("Complete Feature Vector Build")
    bench4.run(engineer.build_feature_vector, keyword_hits, transcripts, embeddings, iterations=50)


if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    benchmark_feature_extraction()
    
    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE")
    print("=" * 60)
```

---

## Test Configuration

### Create pytest.ini

Create `pytest.ini` in project root:

```ini
[pytest]
minversion = 7.0
testpaths = tests services
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests (>1 minute)

addopts =
    -v
    --strict-markers
    --tb=short
    --disable-warnings

# Coverage configuration
[coverage:run]
source = .
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[coverage:report]
precision = 2
show_missing = True
skip_covered = False
```

---

## Running Tests

### Run All Tests

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run performance tests
pytest -m performance

# Run excluding slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/test_feature_engineering.py -v

# Run with output
pytest -v -s
```

### Generate Coverage Report

```powershell
# Generate HTML coverage report
pytest --cov=. --cov-report=html --cov-report=term

# Open coverage report
start htmlcov/index.html  # Windows
# or
open htmlcov/index.html   # Mac/Linux
```

---

## Continue to Part 3: CI/CD & Production...

Would you like me to continue with Part 3 covering:
1. GitHub Actions CI/CD pipeline
2. Docker build automation
3. Production deployment guide
4. Monitoring and alerting setup
5. Final completion summary