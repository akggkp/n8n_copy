from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Service", version="1.0.0")

# Trading Keywords Database - Comprehensive list of trading terms
KEYWORDS_DB = {
    "technical_indicator": [
        "rsi", "relative strength index", 
        "macd", "moving average convergence divergence",
        "bollinger bands", "bollinger",
        "moving average", "ma", "ema", "sma",
        "exponential moving average", "simple moving average",
        "stochastic", "stochastic oscillator",
        "adx", "average directional index",
        "momentum", "momentum indicator",
        "volume", "volume indicator",
        "fibonacci", "fibonacci retracement",
        "atr", "average true range",
        "cci", "commodity channel index",
        "williams %r", "parabolic sar",
        "ichimoku", "ichimoku cloud"
    ],
    "price_action": [
        "support", "support level",
        "resistance", "resistance level",
        "breakout", "breakdown",
        "trend", "uptrend", "downtrend", "sideways",
        "reversal", "trend reversal",
        "consolidation", "consolidation phase",
        "pullback", "retracement",
        "swing high", "swing low",
        "higher high", "higher low",
        "lower high", "lower low",
        "double top", "double bottom",
        "head and shoulders", "inverse head and shoulders",
        "triangle", "ascending triangle", "descending triangle",
        "flag", "pennant", "wedge"
    ],
    "candlestick_pattern": [
        "doji", "hammer", "inverted hammer",
        "engulfing", "bullish engulfing", "bearish engulfing",
        "shooting star", "hanging man",
        "morning star", "evening star",
        "three white soldiers", "three black crows",
        "harami", "bullish harami", "bearish harami",
        "piercing", "dark cloud cover",
        "spinning top", "marubozu"
    ],
    "risk_management": [
        "stop loss", "stop-loss", "sl",
        "take profit", "take-profit", "tp",
        "risk reward", "risk-reward ratio", "r:r",
        "position size", "position sizing",
        "risk management", "money management",
        "drawdown", "maximum drawdown",
        "portfolio", "diversification",
        "leverage", "margin"
    ],
    "order_type": [
        "market order", "limit order",
        "stop order", "stop-limit order",
        "trailing stop", "trailing stop-loss",
        "oco", "one cancels other",
        "bracket order", "conditional order",
        "good till cancelled", "gtc",
        "day order", "fill or kill", "fok"
    ],
    "trading_strategy": [
        "scalping", "day trading", "swing trading",
        "position trading", "trend following",
        "mean reversion", "breakout trading",
        "momentum trading", "range trading",
        "arbitrage", "hedging"
    ],
    "market_structure": [
        "bull market", "bear market",
        "market cycle", "market phase",
        "accumulation", "distribution",
        "markup", "markdown",
        "volatility", "liquidity",
        "bid-ask spread", "order flow"
    ]
}

# Request/Response models
class ConceptRequest(BaseModel):
    transcript: str

class KeywordMatch(BaseModel):
    keyword: str
    category: str
    confidence: float
    context: str
    start_pos: int
    end_pos: int

class ConceptResponse(BaseModel):
    status: str
    keywords: List[KeywordMatch]
    total_found: int

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ml-service",
        "version": "1.0.0",
        "status": "running",
        "keywords_loaded": sum(len(v) for v in KEYWORDS_DB.values())
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ml-service",
        "keywords_loaded": sum(len(v) for v in KEYWORDS_DB.values()),
        "categories": list(KEYWORDS_DB.keys()),
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/extract_concepts", response_model=ConceptResponse)
async def extract_concepts(request: ConceptRequest):
    """
    Extract trading concepts from transcript
    
    Uses rule-based keyword matching with category classification
    
    Args:
        request: ConceptRequest with transcript text
    
    Returns:
        ConceptResponse with detected keywords and metadata
    """
    try:
        transcript = request.transcript
        transcript_lower = transcript.lower()
        
        logger.info(f"Extracting concepts from transcript ({len(transcript)} chars)")
        
        detected_keywords = []
        
        # Iterate through each category and keywords
        for category, keywords in KEYWORDS_DB.items():
            for keyword in keywords:
                # Create regex pattern for whole word matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = list(re.finditer(pattern, transcript_lower, re.IGNORECASE))
                
                for match in matches:
                    # Extract context (50 characters before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(transcript), match.end() + 50)
                    context = transcript[start:end].strip()
                    
                    detected_keywords.append({
                        "keyword": keyword,
                        "category": category,
                        "confidence": 1.0,  # Rule-based = 100% confidence
                        "context": context,
                        "start_pos": match.start(),
                        "end_pos": match.end()
                    })
        
        # Remove duplicates (same keyword at same position)
        unique_keywords = []
        seen = set()
        
        for kw in detected_keywords:
            key = (kw['keyword'], kw['start_pos'])
            if key not in seen:
                seen.add(key)
                unique_keywords.append(kw)
        
        # Sort by position in text
        unique_keywords.sort(key=lambda x: x['start_pos'])
        
        logger.info(f"Found {len(unique_keywords)} unique keywords")
        
        return ConceptResponse(
            status="success",
            keywords=unique_keywords,
            total_found=len(unique_keywords)
        )
    
    except Exception as e:
        logger.error(f"Concept extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_patterns")
async def detect_patterns(frame_data: Dict):
    """
    Detect candlestick patterns in frame (placeholder for future CV model)
    
    Args:
        frame_data: Dictionary with frame information
    
    Returns:
        Dictionary with detected patterns (currently empty)
    """
    logger.info("Pattern detection called (not yet implemented)")
    
    # TODO: Implement actual pattern detection with computer vision model
    return {
        "status": "success",
        "patterns": [],
        "note": "Pattern detection not yet implemented - requires CV model"
    }

@app.get("/categories")
async def get_categories():
    """
    Get available keyword categories
    
    Returns:
        Dictionary with categories and keyword counts
    """
    categories = {}
    for category, keywords in KEYWORDS_DB.items():
        categories[category] = {
            "count": len(keywords),
            "sample_keywords": keywords[:5]
        }
    
    return {
        "status": "success",
        "categories": categories,
        "total_keywords": sum(len(v) for v in KEYWORDS_DB.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)