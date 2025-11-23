# tests/performance/test_load.py
# Load testing for API endpoints

import pytest
import requests
import time
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
            futures = [executor.submit(self.make_request)
                       for _ in range(num_requests)]
            self.results = [f.result()
                            for f in concurrent.futures.as_completed(futures)]

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
