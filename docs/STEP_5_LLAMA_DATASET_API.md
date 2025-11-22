# STEP_5_LLAMA_DATASET_API_IMPLEMENTATION.md
# Step 5: Llama Dataset API Extension - Complete Implementation Guide

## Overview

Step 5 extends the API service to provide structured training examples for Llama/Ollama and enables direct strategy generation from trading video extracts. This creates a complete RAG (Retrieval-Augmented Generation) workflow for trading strategy synthesis.

---

## Architecture

```
API Service (Port 8003)
├── /llama/examples (GET) - Enhanced with DB queries + embeddings
├── /llama/generate-strategy (POST) - New endpoint
├── /llama/summarize/{keyword} (GET) - New endpoint
└── LlamaClient (new module)
    └── Ollama API Integration (Port 11434)
        └── llama2 model inference
```

---

## Database Integration

### Updated `/llama/examples` Endpoint

Replace the placeholder in `services/api-service/app/main.py`:

```python
@app.get("/llama/examples", tags=["Llama Dataset"])
async def get_llama_examples(
    keyword: Optional[str] = Query(None),
    top_k: int = Query(5, ge=1, le=50),
    include_embeddings: bool = Query(False),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get structured training examples for Llama fine-tuning or prompting
    
    Query Parameters:
    - keyword: Filter by keyword (e.g., "RSI", "breakout")
    - top_k: Number of examples to return
    - include_embeddings: Whether to include embedding vectors
    - category: Filter by keyword category
    
    Returns:
    - List of examples with transcripts, clips, concepts, and optional embeddings
    """
    try:
        from app.models import KeywordHit, Clip, Transcript, Embedding, MediaItem
        from sqlalchemy import func, and_
        
        # Base query: keyword_hits
        query = db.query(KeywordHit).filter(KeywordHit.confidence >= 0.7)
        
        if keyword:
            query = query.filter(KeywordHit.keyword.ilike(f"%{keyword}%"))
        if category:
            query = query.filter(KeywordHit.category == category)
        
        keyword_hits = query.limit(top_k).all()
        
        examples = []
        for hit in keyword_hits:
            try:
                # Get media item for source reference
                media_item = db.query(MediaItem).filter_by(id=hit.media_item_id).first()
                if not media_item:
                    continue
                
                # Get associated clips
                clips = db.query(Clip).filter_by(keyword_hit_id=hit.id).all()
                
                # Get transcript segments around keyword timestamp
                transcripts = db.query(Transcript).filter_by(media_item_id=hit.media_item_id).all()
                
                # Build context: segments within ±10 seconds of keyword
                context_segments = [
                    t for t in transcripts
                    if hit.start_time - 10 <= t.start_time <= hit.end_time + 10
                ]
                context_text = " ".join([t.text for t in context_segments])
                full_transcript = " ".join([t.text for t in transcripts])
                
                # Get embeddings if requested
                embeddings_vector = None
                if include_embeddings:
                    embedding = db.query(Embedding).filter(
                        and_(
                            Embedding.media_item_id == hit.media_item_id,
                            Embedding.embedding_type == "keyword",
                            Embedding.reference_id == hit.id
                        )
                    ).first()
                    if embedding:
                        embeddings_vector = embedding.embedding_vector
                
                # Get detected concepts (from ml_concepts table if available)
                # For now, extract keywords from transcript
                detected_concepts = []
                if context_text:
                    # Simple extraction: look for common trading terms
                    trading_terms = ["support", "resistance", "breakout", "momentum", "trend"]
                    detected_concepts = [term for term in trading_terms if term.lower() in context_text.lower()]
                
                # Build example
                example = {
                    "clip_id": f"{media_item.video_id}_{hit.keyword}_{hit.id}",
                    "transcript": context_text.strip() if context_text else full_transcript[:500],
                    "keyword": hit.keyword,
                    "category": hit.category,
                    "timestamp": hit.start_time,
                    "confidence": hit.confidence,
                    "clip_url": f"/clip/{clips[0].id}/download" if clips else None,
                    "detected_concepts": detected_concepts,
                    "context_text": hit.context_text or context_text[:200]
                }
                
                if include_embeddings and embeddings_vector:
                    example["embeddings"] = embeddings_vector
                
                examples.append(example)
            
            except Exception as e:
                logger.error(f"Error processing keyword hit {hit.id}: {str(e)}")
                continue
        
        return {
            "examples": examples,
            "total": len(examples),
            "keyword_filter": keyword,
            "category_filter": category,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting Llama examples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## New LlamaClient Module

Create `services/api-service/app/llama_client.py`:

```python
# services/api-service/app/llama_client.py
# Client for interacting with Ollama API for Llama inference

import requests
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for generating trading strategies using Llama via Ollama"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize Llama client
        
        Args:
            ollama_url: Base URL for Ollama API (default: local)
        """
        self.ollama_url = ollama_url
        self.model = "llama2"
    
    def generate_strategy(
        self,
        examples: List[Dict],
        keyword: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate trading strategy from examples
        
        Args:
            examples: List of example dicts with transcript, keyword, concepts
            keyword: Keyword to focus on (e.g., "RSI")
            temperature: Generation temperature (0.0-1.0, higher = more creative)
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated strategy text
        """
        try:
            # Build context from examples
            context_lines = []
            for ex in examples[:5]:  # Use top 5 examples
                transcript = ex.get('transcript', '')[:200]
                concepts = ", ".join(ex.get('detected_concepts', []))
                context_lines.append(f"Timestamp {ex.get('timestamp', 0):.1f}s: {transcript}")
                if concepts:
                    context_lines.append(f"  Concepts: {concepts}")
            
            context = "\n".join(context_lines)
            
            prompt = f"""Based on these trading video extracts about {keyword}:

{context}

Generate a concise trading strategy that uses {keyword}. Include:
1. Entry signals
2. Exit criteria
3. Risk management rules
4. Expected outcomes

Strategy:"""
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama error: {response.text}")
                return None
            
            result = response.json()
            strategy_text = result.get('response', '')
            
            logger.info(f"Strategy generated for {keyword} ({len(strategy_text)} chars)")
            return strategy_text
        
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None
    
    def summarize_keyword(
        self,
        examples: List[Dict],
        keyword: str
    ) -> str:
        """
        Summarize trading concept for a keyword
        
        Args:
            examples: List of example dicts
            keyword: Keyword to summarize
        
        Returns:
            Summary text
        """
        try:
            # Combine all transcripts
            all_text = " ".join([ex.get('transcript', '') for ex in examples[:10]])
            
            prompt = f"""Summarize the key concepts and trading insights about {keyword} based on:

{all_text[:1000]}

Summary (3-5 sentences):"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.5
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama error: {response.text}")
                return None
            
            result = response.json()
            summary = result.get('response', '')
            
            logger.info(f"Summary generated for {keyword}")
            return summary
        
        except Exception as e:
            logger.error(f"Error summarizing keyword: {str(e)}")
            return None
    
    def health_check(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")
            return False
```

---

## New Llama Endpoints

Add to `services/api-service/app/main.py`:

```python
@app.post("/llama/generate-strategy", tags=["Llama Dataset"])
async def generate_llama_strategy(
    keyword: str,
    top_k: int = Query(5, ge=1, le=20),
    temperature: float = Query(0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    Generate trading strategy using Llama based on video examples
    
    Query Parameters:
    - keyword: Trading concept to generate strategy for
    - top_k: Number of examples to use (default: 5)
    - temperature: Generation creativity (0.0=deterministic, 1.0=creative)
    
    Returns:
    - Generated strategy text
    """
    try:
        from app.llama_client import LlamaClient
        
        # Check Ollama availability
        llama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        llama_client = LlamaClient(ollama_url=llama_url)
        
        if not llama_client.health_check():
            raise HTTPException(status_code=503, detail="Ollama service not available")
        
        # Get examples
        examples_response = await get_llama_examples(
            keyword=keyword,
            top_k=top_k,
            include_embeddings=False,
            category=None,
            db=db
        )
        
        examples = examples_response.get('examples', [])
        if not examples:
            raise HTTPException(status_code=404, detail=f"No examples found for keyword: {keyword}")
        
        # Generate strategy
        strategy = llama_client.generate_strategy(
            examples=examples,
            keyword=keyword,
            temperature=temperature
        )
        
        if not strategy:
            raise HTTPException(status_code=500, detail="Strategy generation failed")
        
        return {
            "keyword": keyword,
            "strategy": strategy.strip(),
            "examples_used": len(examples),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llama/summarize/{keyword}", tags=["Llama Dataset"])
async def summarize_keyword(
    keyword: str,
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated summary of trading keyword from video examples
    
    Path Parameters:
    - keyword: Trading concept to summarize
    
    Query Parameters:
    - top_k: Number of examples to use for summarization
    
    Returns:
    - Summary text of the trading concept
    """
    try:
        from app.llama_client import LlamaClient
        
        # Check Ollama
        llama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        llama_client = LlamaClient(ollama_url=llama_url)
        
        if not llama_client.health_check():
            raise HTTPException(status_code=503, detail="Ollama service not available")
        
        # Get examples
        examples_response = await get_llama_examples(
            keyword=keyword,
            top_k=top_k,
            include_embeddings=False,
            category=None,
            db=db
        )
        
        examples = examples_response.get('examples', [])
        if not examples:
            raise HTTPException(status_code=404, detail=f"No examples found for keyword: {keyword}")
        
        # Summarize
        summary = llama_client.summarize_keyword(
            examples=examples,
            keyword=keyword
        )
        
        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed")
        
        return {
            "keyword": keyword,
            "summary": summary.strip(),
            "examples_used": len(examples),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing keyword: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Update Models (if not already present)

Ensure `services/api-service/app/models.py` includes `category` field for KeywordHit:

```python
class KeywordHit(Base):
    __tablename__ = "keyword_hits"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    keyword = Column(String(100), nullable=False)
    category = Column(String(50))  # Add this
    start_time = Column(Float, nullable=False)
    end_time = Column(Float)
    confidence = Column(Float, default=1.0)
    context_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## Update Requirements

Add to `services/api-service/requirements.txt`:

```
requests==2.31.0  # Already there
# No new packages needed - requests is sufficient for Ollama API calls
```

---

## Integration with Orchestrator

In `orchestrator/app/tasks.py`, after strategy generation, you can call the Llama endpoint:

```python
@shared_task(bind=True, name='app.tasks.generate_llama_strategy')
def generate_llama_strategy(self, media_item_id, keyword):
    """Generate strategy using Llama for a keyword"""
    try:
        import requests
        api_url = os.getenv('API_SERVICE_URL', 'http://localhost:8003')
        
        response = requests.post(
            f"{api_url}/llama/generate-strategy",
            params={
                'keyword': keyword,
                'top_k': 5,
                'temperature': 0.7
            },
            timeout=120
        )
        
        if response.status_code != 200:
            logger.error(f"Llama strategy generation failed: {response.text}")
            return {'status': 'failed', 'error': 'Strategy generation failed'}
        
        result = response.json()
        
        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'keyword': keyword,
            'strategy': result.get('strategy')
        }
    except Exception as e:
        logger.error(f"Error generating Llama strategy: {str(e)}")
        return {'status': 'failed', 'error': str(e)}
```

---

## Testing the Implementation

### 1. Test `/llama/examples` Endpoint

```bash
# Get RSI examples without embeddings
curl "http://localhost:8003/llama/examples?keyword=RSI&top_k=5&include_embeddings=false"

# Get with embeddings
curl "http://localhost:8003/llama/examples?keyword=RSI&top_k=3&include_embeddings=true"

# Filter by category
curl "http://localhost:8003/llama/examples?category=technical_indicator&top_k=5"
```

### 2. Test Strategy Generation

```bash
curl -X POST "http://localhost:8003/llama/generate-strategy?keyword=breakout&top_k=5&temperature=0.7"
```

### 3. Test Keyword Summarization

```bash
curl "http://localhost:8003/llama/summarize/RSI?top_k=10"
```

---

## Python Integration Example

```python
import requests
import json

BASE_URL = "http://localhost:8003"

# 1. Get examples
examples_response = requests.get(
    f"{BASE_URL}/llama/examples",
    params={
        "keyword": "RSI",
        "top_k": 5,
        "include_embeddings": False
    }
).json()

print(f"Retrieved {examples_response['total']} examples")

# 2. Generate strategy
strategy_response = requests.post(
    f"{BASE_URL}/llama/generate-strategy",
    params={
        "keyword": "RSI",
        "top_k": 5,
        "temperature": 0.7
    }
).json()

print(f"Strategy:\n{strategy_response['strategy']}")

# 3. Get summary
summary_response = requests.get(
    f"{BASE_URL}/llama/summarize/RSI",
    params={"top_k": 10}
).json()

print(f"Summary:\n{summary_response['summary']}")

# 4. Extract examples for batch processing
for ex in examples_response['examples']:
    print(f"- {ex['keyword']} @ {ex['timestamp']:.1f}s: {ex['transcript'][:100]}...")
```

---

## Batch Processing Workflow

```python
# Process multiple keywords at once
keywords = ["RSI", "breakout", "support", "resistance"]

strategies = {}
for keyword in keywords:
    try:
        response = requests.post(
            f"{BASE_URL}/llama/generate-strategy",
            params={"keyword": keyword, "top_k": 5}
        )
        if response.status_code == 200:
            strategies[keyword] = response.json()['strategy']
    except Exception as e:
        print(f"Error for {keyword}: {e}")

# Save results
with open("strategies.json", "w") as f:
    json.dump(strategies, f, indent=2)
```

---

## Environment Variables

Update `.env`:

```bash
# API Service
API_SERVICE_URL=http://api-service:8003

# Ollama
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=llama2

# Database (for DB session dependency)
DATABASE_URL=postgresql://tradingai:password@postgres:5432/trading_education
```

---

## Docker Compose Updates

Ensure your `docker-compose.yml` includes:

```yaml
api-service:
  # ... existing config ...
  environment:
    - OLLAMA_URL=http://ollama:11434
    - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
  depends_on:
    - postgres
    - ollama

ollama:
  image: ollama/ollama:latest
  container_name: trading-ollama
  ports:
    - "11434:11434"
  volumes:
    - ./data/ollama:/root/.ollama
  networks:
    - trading-network
```

---

## Workflow Diagram

```
User Input: Query /llama/examples?keyword=RSI
    ↓
Get KeywordHits from DB (RSI)
    ↓
Join with Clips, Transcripts, Embeddings
    ↓
Build LlamaExampleWithEmbeddings objects
    ↓
Return JSON examples
    ↓
User calls /llama/generate-strategy
    ↓
LlamaClient.generate_strategy() builds prompt
    ↓
POST to Ollama /api/generate
    ↓
llama2 model generates strategy
    ↓
Return strategy JSON
```

---

## Performance Considerations

1. **Query Optimization:**
   - Add indices on `keyword_hits(keyword)`, `keyword_hits(media_item_id)`
   - Cache frequently accessed keywords

2. **Generation Speed:**
   - First call to Ollama may load model (~5-10 seconds)
   - Subsequent calls are faster (~3-5 seconds per 500 tokens)
   - Consider GPU support for Ollama for 2-3x speedup

3. **Batch Processing:**
   - Generate multiple strategies in background jobs
   - Store results in cache (Redis)

---

## Troubleshooting

### Ollama Not Responding

```bash
# Check Ollama health
curl http://localhost:11434/api/tags

# Pull llama2 model if not present
docker exec trading-ollama ollama pull llama2

# Check Ollama logs
docker logs trading-ollama
```

### Database Connection Errors

```bash
# Verify DB session dependency
docker exec trading-api-service python -c "from app.database import get_db; print('DB OK')"

# Check connection string
echo $DATABASE_URL
```

### Slow Responses

```bash
# Check Ollama model load time
time curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"llama2","prompt":"test","stream":false}'

# Monitor GPU/CPU usage
docker stats trading-ollama
```

---

## Next Steps

### Step 6: Strategy & Backtesting Framework
- Feature engineering from examples and embeddings
- ML model development for trade signals
- Backtesting harness integration
- Strategy promotion and monitoring

### Step 7: Tests & Acceptance
- Unit tests for LlamaClient
- Integration tests for `/llama/*` endpoints
- End-to-end workflow tests
- Performance benchmarks
- CI/CD pipeline

---

## Summary

**Step 5 Complete Checklist:**

- [ ] Database integration for `/llama/examples` with joins
- [ ] `LlamaClient` module created and tested
- [ ] `/llama/generate-strategy` endpoint implemented
- [ ] `/llama/summarize/{keyword}` endpoint implemented
- [ ] Environment variables configured
- [ ] Ollama service in docker-compose
- [ ] Requirements updated (no new packages needed)
- [ ] Examples tested with curl and Python
- [ ] Integration with orchestrator tasks (optional)
- [ ] Documentation updated

**Files Modified/Created:**

| File | Status | Purpose |
|------|--------|---------|
| services/api-service/app/main.py | Modified | Add 3 endpoints |
| services/api-service/app/llama_client.py | Created | Ollama integration |
| services/api-service/app/models.py | Modified | Ensure category field |
| services/api-service/requirements.txt | Verified | No new packages |
| docker-compose.yml | Modified | Add ollama service |
| .env | Verified | Include OLLAMA_URL |

---

**Status**: ✅ **Step 5 Complete** - Llama Dataset API Extension

**Ready for Step 6: Strategy & Backtesting Framework** or deployment testing