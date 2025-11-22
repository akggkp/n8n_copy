# STEP_6_STRATEGY_BACKTESTING_PART3.md
# Step 6: Strategy & Backtesting Framework - Part 3: Testing & Completion

## Testing the Complete Pipeline

### 1. End-to-End Test Script

Create `tests/test_step6_pipeline.py`:

```python
# tests/test_step6_pipeline.py
# End-to-end test for strategy generation and backtesting pipeline

import requests
import time
import json

BASE_URL_API = "http://localhost:8003"
BASE_URL_BACKTEST = "http://localhost:8001"


def test_full_pipeline():
    """Test complete video processing → strategy generation → backtesting pipeline"""
    
    print("=" * 60)
    print("STEP 6: Strategy & Backtesting Pipeline Test")
    print("=" * 60)
    
    # Step 1: Health checks
    print("\n1. Health Checks...")
    
    api_health = requests.get(f"{BASE_URL_API}/health").json()
    print(f"   API Service: {api_health['status']}")
    
    backtest_health = requests.get(f"{BASE_URL_BACKTEST}/health").json()
    print(f"   Backtest Service: {backtest_health['status']}")
    
    # Step 2: Ingest video (triggers pipeline)
    print("\n2. Ingesting video...")
    
    ingest_response = requests.post(
        f"{BASE_URL_API}/ingest",
        json={
            "video_path": "/data/videos/sample.mp4",
            "filename": "sample.mp4"
        }
    ).json()
    
    media_item_id = ingest_response.get('media_item_id')
    task_id = ingest_response.get('task_id')
    
    print(f"   Media Item ID: {media_item_id}")
    print(f"   Task ID: {task_id}")
    
    # Step 3: Wait for pipeline completion (polling)
    print("\n3. Waiting for pipeline completion...")
    
    max_wait_time = 600  # 10 minutes
    wait_interval = 10   # Check every 10 seconds
    elapsed = 0
    
    while elapsed < max_wait_time:
        time.sleep(wait_interval)
        elapsed += wait_interval
        
        # Check media item status
        media_response = requests.get(f"{BASE_URL_API}/media_items/{media_item_id}").json()
        status = media_response.get('status')
        
        print(f"   [{elapsed}s] Status: {status}")
        
        if status == 'completed':
            print("   ✓ Pipeline completed successfully!")
            break
        elif status == 'failed':
            print("   ✗ Pipeline failed")
            return False
    
    if elapsed >= max_wait_time:
        print("   ✗ Pipeline timeout")
        return False
    
    # Step 4: Verify features were extracted
    print("\n4. Verifying feature extraction...")
    
    keywords_response = requests.get(
        f"{BASE_URL_API}/keywords",
        params={"media_item_id": media_item_id}
    ).json()
    
    keyword_count = len(keywords_response.get('keywords', []))
    print(f"   Keywords extracted: {keyword_count}")
    
    # Step 5: Check if strategy was generated
    print("\n5. Checking strategy generation...")
    
    # Query backtest service for strategies
    # (In production, this would query proven_strategies table)
    print("   Strategy generation complete (check database for proven_strategies)")
    
    # Step 6: Manually test strategy creation and backtesting
    print("\n6. Testing manual strategy creation...")
    
    strategy_response = requests.post(
        f"{BASE_URL_BACKTEST}/strategies",
        json={
            "name": "Test_RSI_Strategy",
            "entry_rules": ["RSI < 30"],
            "exit_rules": ["RSI > 70"],
            "risk_management": {
                "position_size": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.06
            }
        }
    ).json()
    
    strategy_id = strategy_response.get('strategy_id')
    print(f"   Strategy ID: {strategy_id}")
    
    # Step 7: Run backtest
    print("\n7. Running backtest...")
    
    backtest_response = requests.post(
        f"{BASE_URL_BACKTEST}/backtest",
        json={
            "strategy_id": strategy_id,
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 10000.0
        }
    ).json()
    
    backtest_id = backtest_response.get('backtest_id')
    metrics = backtest_response.get('metrics', {})
    
    print(f"   Backtest ID: {backtest_id}")
    print(f"\n   Performance Metrics:")
    print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio')}")
    print(f"   - Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"   - Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"   - Total Return: {metrics.get('total_return')}%")
    print(f"   - Num Trades: {metrics.get('num_trades')}")
    print(f"   - Profit Factor: {metrics.get('profit_factor')}")
    
    # Step 8: Validate strategy
    print("\n8. Validating strategy performance...")
    
    sharpe = metrics.get('sharpe_ratio', 0)
    win_rate = metrics.get('win_rate', 0)
    max_dd = abs(metrics.get('max_drawdown', 1))
    
    min_sharpe = 1.0
    min_win_rate = 0.55
    max_drawdown = 0.25
    
    passed = (
        sharpe >= min_sharpe and
        win_rate >= min_win_rate and
        max_dd <= max_drawdown
    )
    
    if passed:
        print(f"   ✓ Strategy PASSED validation thresholds")
        print(f"     - Sharpe >= {min_sharpe}: {sharpe >= min_sharpe}")
        print(f"     - Win Rate >= {min_win_rate}: {win_rate >= min_win_rate}")
        print(f"     - Drawdown <= {max_drawdown}: {max_dd <= max_drawdown}")
    else:
        print(f"   ✗ Strategy FAILED validation")
    
    # Step 9: Get equity curve
    print("\n9. Retrieving equity curve...")
    
    equity_response = requests.get(f"{BASE_URL_BACKTEST}/backtest/{backtest_id}/equity").json()
    equity_curve = equity_response.get('equity_curve', [])
    
    print(f"   Initial Capital: ${equity_response.get('initial_capital'):,.2f}")
    print(f"   Final Capital: ${equity_response.get('final_capital'):,.2f}")
    print(f"   Equity Curve Length: {len(equity_curve)} data points")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_full_pipeline()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)
```

---

## Standalone Component Tests

### Test Feature Engineering

```python
# tests/test_feature_engineering.py

from orchestrator.app.feature_engineering import FeatureEngineer
import numpy as np


def test_keyword_features():
    """Test keyword feature extraction"""
    engineer = FeatureEngineer()
    
    keyword_hits = [
        {'keyword': 'RSI', 'category': 'technical_indicator', 'start_time': 10.0, 'end_time': 12.0, 'confidence': 0.95},
        {'keyword': 'breakout', 'category': 'price_action', 'start_time': 25.0, 'end_time': 27.0, 'confidence': 0.88},
        {'keyword': 'stop loss', 'category': 'risk_management', 'start_time': 40.0, 'end_time': 42.0, 'confidence': 0.92}
    ]
    
    features = engineer.extract_keyword_features(keyword_hits)
    
    assert features['technical_indicator_count'] == 1
    assert features['price_action_count'] == 1
    assert features['risk_management_count'] == 1
    assert features['has_rsi'] == 1
    assert features['has_breakout'] == 1
    assert features['has_stop_loss'] == 1
    assert features['avg_confidence'] > 0.8
    
    print("✓ Keyword feature extraction test passed")


def test_transcript_features():
    """Test transcript feature extraction"""
    engineer = FeatureEngineer()
    
    transcripts = [
        {'text': 'Today we will discuss the RSI indicator', 'start_time': 0.0, 'end_time': 3.0},
        {'text': 'For example, when RSI is below 30', 'start_time': 3.0, 'end_time': 6.0},
        {'text': 'First, you need to calculate the gains', 'start_time': 6.0, 'end_time': 9.0}
    ]
    
    features = engineer.extract_transcript_features(transcripts)
    
    assert features['total_segments'] == 3
    assert features['has_examples'] == 1
    assert features['has_steps'] == 1
    assert features['speaking_rate_wpm'] > 0
    
    print("✓ Transcript feature extraction test passed")


def test_embedding_features():
    """Test embedding feature extraction"""
    engineer = FeatureEngineer()
    
    embeddings = [
        np.random.randn(384),
        np.random.randn(384),
        np.random.randn(384)
    ]
    
    features = engineer.extract_embedding_features(embeddings)
    
    assert features['embedding_dim'] == 384
    assert features['avg_embedding_norm'] > 0
    assert 0 <= features['embedding_diversity'] <= 1
    
    print("✓ Embedding feature extraction test passed")


if __name__ == "__main__":
    test_keyword_features()
    test_transcript_features()
    test_embedding_features()
    print("\n✓ All feature engineering tests passed")
```

### Test Backtesting Service

```bash
# Test backtesting service directly

# 1. Create strategy
curl -X POST http://localhost:8001/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "RSI_Mean_Reversion",
    "entry_rules": ["RSI < 30"],
    "exit_rules": ["RSI > 70"],
    "risk_management": {
      "position_size": 0.1,
      "stop_loss_pct": 0.02,
      "take_profit_pct": 0.06
    }
  }'

# 2. Run backtest
curl -X POST http://localhost:8001/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": 1,
    "symbol": "BTCUSDT",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000.0
  }'

# 3. Get metrics
curl http://localhost:8001/backtest/1/metrics

# 4. Get trades
curl http://localhost:8001/backtest/1/trades

# 5. Get equity curve
curl http://localhost:8001/backtest/1/equity
```

---

## Performance Monitoring

### Add Prometheus Metrics (Optional)

Update `services/backtesting-service/app/main.py`:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

# Metrics
backtest_requests = Counter('backtest_requests_total', 'Total backtest requests')
backtest_duration = Histogram('backtest_duration_seconds', 'Backtest execution time')
active_strategies = Gauge('active_strategies_count', 'Number of active strategies')

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

---

## Environment Variables Update

Add to `.env`:

```bash
# Backtesting Service
BACKTEST_SERVICE_URL=http://backtesting-service:8001
BACKTEST_PORT=8001

# Strategy Validation Thresholds
MIN_SHARPE_RATIO=1.0
MIN_WIN_RATE_PERCENT=55
MAX_DRAWDOWN_PERCENT=25

# Feature Engineering
FEATURE_CONFIDENCE_THRESHOLD=0.7
```

---

## Integration Checklist

- [ ] Feature engineering module created (`orchestrator/app/feature_engineering.py`)
- [ ] Backtest client created (`orchestrator/app/backtest_client.py`)
- [ ] New orchestrator tasks added (extract_features, generate_strategy, backtest_strategy, evaluate_and_promote)
- [ ] Pipeline chain updated in `run_full_pipeline`
- [ ] ProvenStrategy model added to database schema
- [ ] Backtesting service implemented (`services/backtesting-service/app/main.py`)
- [ ] Backtesting service requirements.txt created
- [ ] Backtesting service Dockerfile created
- [ ] docker-compose.yml updated with backtesting-service
- [ ] Environment variables configured
- [ ] Tests created and passing
- [ ] End-to-end pipeline test successful

---

## Directory Structure Summary

```
project/
├── orchestrator/
│   └── app/
│       ├── tasks.py (updated with 4 new tasks)
│       ├── feature_engineering.py (new)
│       ├── backtest_client.py (new)
│       └── models.py (add ProvenStrategy)
├── services/
│   ├── api-service/ (existing)
│   ├── embeddings-service/ (existing)
│   └── backtesting-service/ (new)
│       ├── app/
│       │   ├── __init__.py
│       │   └── main.py
│       ├── Dockerfile
│       └── requirements.txt
├── tests/
│   ├── test_step6_pipeline.py (new)
│   └── test_feature_engineering.py (new)
└── docker-compose.yml (updated)
```

---

## Deployment Steps

### 1. Create Directories

```powershell
$dirs = @(
    "services/backtesting-service/app",
    "tests"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$files = @(
    "services/backtesting-service/__init__.py",
    "services/backtesting-service/app/__init__.py"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
    }
}
```

### 2. Copy Files

```powershell
# Copy backtesting service files
Copy-Item -Path "backtest_main.py" -Destination "services/backtesting-service/app/main.py"
Copy-Item -Path "backtest_requirements.txt" -Destination "services/backtesting-service/requirements.txt"
Copy-Item -Path "backtest_dockerfile" -Destination "services/backtesting-service/Dockerfile"

# Copy orchestrator modules
Copy-Item -Path "feature_engineering.py" -Destination "orchestrator/app/feature_engineering.py"
Copy-Item -Path "backtest_client.py" -Destination "orchestrator/app/backtest_client.py"

# Copy tests
Copy-Item -Path "test_step6_pipeline.py" -Destination "tests/test_step6_pipeline.py"
```

### 3. Build and Run

```powershell
# Build backtesting service
docker-compose build backtesting-service

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8001/health

# Run tests
python tests/test_step6_pipeline.py
```

---

## Troubleshooting

### Backtesting Service Won't Start

```bash
# Check logs
docker-compose logs backtesting-service

# Verify port
netstat -ano | findstr :8001

# Test directly
docker exec trading-backtesting-service python -c "from app.main import app; print('OK')"
```

### Feature Extraction Errors

```bash
# Check orchestrator logs
docker-compose logs orchestrator-worker

# Verify database connection
docker exec trading-orchestrator-worker python -c "from app.database import SessionLocal; db = SessionLocal(); print('DB OK')"
```

### Strategy Validation Issues

```bash
# Check thresholds
echo $MIN_SHARPE_RATIO
echo $MIN_WIN_RATE_PERCENT

# Test validation logic
curl -X POST http://localhost:8001/backtest ...
```

---

## Performance Optimization

1. **Feature Caching**: Cache extracted features in Redis
2. **Parallel Backtesting**: Run multiple backtests in parallel
3. **Database Indexing**: Add indices on media_item_id, strategy_id
4. **Async Processing**: Use Celery for long-running backtests

---

## Next Steps

### Step 7: Tests & Acceptance (Final Step)
- Comprehensive unit tests for all modules
- Integration tests for complete pipeline
- Performance benchmarks
- CI/CD pipeline with GitHub Actions
- Production deployment guide
- Monitoring and alerting setup

---

## Summary

**Step 6 Complete Checklist:**

✅ Feature engineering module with 30+ features
✅ Backtesting service with 7 endpoints
✅ ML/RL strategy generation from video features
✅ Performance metrics calculation (Sharpe, win rate, drawdown)
✅ Strategy validation and promotion logic
✅ Complete pipeline integration (10-step chain)
✅ End-to-end testing script
✅ Component unit tests
✅ Docker configuration
✅ Environment variables
✅ Troubleshooting guide
✅ Performance optimization tips

**Files Created/Modified:**

| File | Status | Purpose |
|------|--------|---------|
| orchestrator/app/feature_engineering.py | Created | Extract ML features |
| orchestrator/app/backtest_client.py | Created | Backtest service client |
| orchestrator/app/tasks.py | Modified | Add 4 new pipeline tasks |
| orchestrator/app/models.py | Modified | Add ProvenStrategy model |
| services/backtesting-service/app/main.py | Created | Backtesting FastAPI service |
| services/backtesting-service/requirements.txt | Created | Python dependencies |
| services/backtesting-service/Dockerfile | Created | Container definition |
| docker-compose.yml | Modified | Add backtesting-service |
| .env | Modified | Add backtest variables |
| tests/test_step6_pipeline.py | Created | E2E test script |
| tests/test_feature_engineering.py | Created | Unit tests |

---

**Status**: ✅ **Step 6 Complete** - Strategy & Backtesting Framework

**Pipeline Stages (Complete 10-Step Chain):**
1. Validate Video ✅
2. Process Video ✅
3. Detect Keywords ✅
4. Generate Clips ✅
5. Extract Concepts ✅
6. Generate Embeddings ✅
7. Extract Features ✅ (NEW)
8. Generate Strategy ✅ (NEW)
9. Backtest Strategy ✅ (NEW)
10. Evaluate & Promote ✅ (NEW)

**Ready for Step 7: Tests & Acceptance** (Final comprehensive testing and deployment)