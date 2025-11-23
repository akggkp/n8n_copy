"""
ML Training Service - Strategy generation and concept extraction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Trading Education AI - ML Service",
    description="Strategy generation and concept extraction API",
    version="1.0.0"
)

# Models
class ConceptExtractionRequest(BaseModel):
    video_id: str
    transcription: str
    detected_charts: List[Dict[str, Any]] = []

class ConceptExtractionResponse(BaseModel):
    status: str
    video_id: str
    concepts: List[str]
    indicators: List[str]
    patterns: List[str]
    message: str

class StrategyGenerationRequest(BaseModel):
    video_id: str
    concepts: List[str]
    indicators: List[str]
    patterns: List[str]

class StrategyGenerationResponse(BaseModel):
    status: str
    video_id: str
    strategies: List[Dict[str, Any]]
    message: str

# Routes
@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "ml-service",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/extract-concepts", response_model=ConceptExtractionResponse)
async def extract_concepts(request: ConceptExtractionRequest):
    """
    Extract trading concepts from transcription
    """
    try:
        logger.info(f"Extracting concepts for video: {request.video_id}")
        
        from app.concept_extractor.extractor import ConceptExtractor
        
        extractor = ConceptExtractor()
        
        concepts = extractor.extract_trading_concepts(request.transcription)
        indicators = extractor.extract_indicators(request.transcription)
        patterns = extractor.extract_patterns(request.transcription, request.detected_charts)
        
        return ConceptExtractionResponse(
            status="success",
            video_id=request.video_id,
            concepts=concepts,
            indicators=indicators,
            patterns=patterns,
            message="Concepts extracted successfully"
        )
    
    except Exception as e:
        logger.error(f"Error extracting concepts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-strategies", response_model=StrategyGenerationResponse)
async def generate_strategies(request: StrategyGenerationRequest):
    """
    Generate trading strategies from concepts
    """
    try:
        logger.info(f"Generating strategies for video: {request.video_id}")
        
        from app.strategy_generator.model_trainer import StrategyGenerator
        
        generator = StrategyGenerator()
        
        strategies = generator.generate_strategies(
            video_id=request.video_id,
            concepts=request.concepts,
            indicators=request.indicators,
            patterns=request.patterns
        )
        
        return StrategyGenerationResponse(
            status="success",
            video_id=request.video_id,
            strategies=strategies,
            message=f"Generated {len(strategies)} strategies"
        )
    
    except Exception as e:
        logger.error(f"Error generating strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/trending")
async def get_trending_strategies(limit: int = 10):
    """Get trending strategies based on backtest performance"""
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        query = text("""
            SELECT 
                s.strategy_id,
                s.strategy_name,
                b.win_rate,
                b.profit_factor,
                COUNT(*) as frequency
            FROM ml_strategies s
            JOIN backtest_results b ON s.strategy_id = b.strategy_id
            WHERE b.is_profitable = 1
            GROUP BY s.strategy_id, s.strategy_name, b.win_rate, b.profit_factor
            ORDER BY b.win_rate DESC, frequency DESC
            LIMIT :limit
        """)
        
        results = session.execute(query, {"limit": limit}).fetchall()
        session.close()
        
        return {
            "status": "success",
            "count": len(results),
            "strategies": [
                {
                    "strategy_id": r[0],
                    "name": r[1],
                    "win_rate": float(r[2]),
                    "profit_factor": float(r[3]),
                    "frequency": r[4]
                }
                for r in results
            ]
        }
    
    except Exception as e:
        logger.error(f"Error fetching trending strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
