"""
Backtesting Service API - Trading Education AI
REST API for strategy backtesting and validation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trading Education AI - Backtesting Service",
    description="Strategy backtesting and validation API",
    version="1.0.0"
)

# Models
class BacktestRequest(BaseModel):
    video_id: str
    strategy: Dict[str, Any]
    symbol: str = "NIFTY"
    timeframe: str = "15m"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"

class BacktestResponse(BaseModel):
    status: str
    video_id: str
    backtest_results: Dict[str, Any]
    message: str

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "backtesting",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/backtest", response_model=BacktestResponse)
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest a trading strategy
    
    Args:
        request: BacktestRequest with strategy details
        
    Returns:
        BacktestResponse with results
    """
    try:
        logger.info(f"Backtesting strategy for video: {request.video_id}")
        
        # Import backtester engine
        from app.engine.backtester import BacktestEngine
        
        engine = BacktestEngine()
        
        # Run backtest
        is_profitable = engine.backtest_strategy(
            strategy_id=request.video_id,
            strategy_data=request.strategy,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Get results from database
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        db_engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=db_engine)
        session = Session()
        
        # Query backtest results
        query = text("""
            SELECT win_rate, profit_factor, sharpe_ratio, total_pnl, 
                   total_trades, winning_trades, losing_trades 
            FROM backtest_results 
            WHERE strategy_id = :sid 
            ORDER BY tested_at DESC LIMIT 1
        """)
        
        result = session.execute(query, {"sid": request.video_id}).fetchone()
        session.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest results not found")
        
        return BacktestResponse(
            status="success",
            video_id=request.video_id,
            backtest_results={
                "is_profitable": is_profitable,
                "win_rate": float(result[0]) if result[0] else 0,
                "profit_factor": float(result[1]) if result[1] else 0,
                "sharpe_ratio": float(result[2]) if result[2] else 0,
                "total_pnl": float(result[3]) if result[3] else 0,
                "total_trades": int(result[4]) if result[4] else 0,
                "winning_trades": int(result[5]) if result[5] else 0,
                "losing_trades": int(result[6]) if result[6] else 0
            },
            message="Backtest completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error backtesting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/profitable")
async def get_profitable_strategies():
    """Get all profitable strategies"""
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        db_engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=db_engine)
        session = Session()
        
        query = text("""
            SELECT strategy_id, win_rate, profit_factor, sharpe_ratio, total_pnl
            FROM backtest_results
            WHERE win_rate > :min_wr 
            AND profit_factor > :min_pf
            AND sharpe_ratio > :min_sr
            ORDER BY win_rate DESC
        """)
        
        min_wr = float(os.getenv("MIN_WIN_RATE_TO_SAVE", 55))
        min_pf = float(os.getenv("MIN_PROFIT_FACTOR", 1.5))
        min_sr = float(os.getenv("MIN_SHARPE_RATIO", 0.5))
        
        results = session.execute(
            query, 
            {"min_wr": min_wr, "min_pf": min_pf, "min_sr": min_sr}
        ).fetchall()
        
        session.close()
        
        return {
            "status": "success",
            "count": len(results),
            "strategies": [
                {
                    "strategy_id": r[0],
                    "win_rate": float(r[1]),
                    "profit_factor": float(r[2]),
                    "sharpe_ratio": float(r[3]),
                    "total_pnl": float(r[4])
                }
                for r in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
