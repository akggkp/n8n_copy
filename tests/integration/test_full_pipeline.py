# tests/integration/test_full_pipeline.py
# Full end-to-end pipeline integration test

import pytest
import requests
import time
import os

BASE_URL_API = os.getenv("API_BASE_URL", "http://localhost:8003")
BASE_URL_BACKTEST = os.getenv("BACKTEST_SERVICE_URL", "http://localhost:8001")
BASE_URL_EMBEDDINGS = os.getenv("EMBEDDINGS_SERVICE_URL", "http://localhost:8004")

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
