import requests
import time
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8003")


def test_llama_pipeline():
    """Test complete Llama integration pipeline"""

    print("\n--- Starting Llama Integration Tests ---")

    # Step 1: Test /llama/examples endpoint
    print("Step 1: Testing /llama/examples endpoint...")
    examples_response = requests.get(
        f"{API_BASE}/llama/examples",
        params={"keyword": "RSI", "top_k": 3},
        timeout=30
    )
    assert examples_response.status_code == 200, f"Expected 200, got {examples_response.status_code}: {examples_response.text}"
    examples = examples_response.json()
    assert "examples" in examples and len(
        examples["examples"]) > 0, "No examples found for RSI"
    print(f"✓ Found {examples['total']} examples for 'RSI'")
    time.sleep(1)  # Small delay

    # Step 2: Test /llama/generate-strategy endpoint
    print("\nStep 2: Testing /llama/generate-strategy endpoint...")
    strategy_response = requests.post(
        f"{API_BASE}/llama/generate-strategy",
        params={"keyword": "RSI", "top_k": 3, "temperature": 0.7},
        timeout=180  # Ollama can be slow on first call
    )
    assert strategy_response.status_code == 200, f"Expected 200, got {strategy_response.status_code}: {strategy_response.text}"
    strategy = strategy_response.json()
    assert "strategy" in strategy and len(
        strategy["strategy"]) > 50, "Generated strategy too short or missing"
    print(f"✓ Generated strategy (length: {len(strategy['strategy'])} chars)")
    time.sleep(1)  # Small delay

    # Step 3: Test /llama/summarize endpoint
    print("\nStep 3: Testing /llama/summarize endpoint...")
    summary_response = requests.get(
        f"{API_BASE}/llama/summarize/RSI",
        params={"top_k": 2},
        timeout=90
    )
    assert summary_response.status_code == 200, f"Expected 200, got {summary_response.status_code}: {summary_response.text}"
    summary = summary_response.json()
    assert "summary" in summary and len(
        summary["summary"]) > 20, "Generated summary too short or missing"
    print(f"✓ Generated summary: {summary['summary'][:100]}...")
    time.sleep(1)  # Small delay

    print("\nAll Llama integration tests passed!")


if __name__ == "__main__":
    # To run this test:
    # 1. Ensure Docker Compose services are up: `docker compose up -d`
    # 2. Ensure Ollama 'llama2' model is pulled: `docker exec trading-ollama ollama pull llama2`
    # 3. Run this script using pytest or directly: `python -m pytest tests/test_llama_integration.py`
    # or `python tests/test_llama_integration.py` (if it contains direct
    # executable logic)
    test_llama_pipeline()
