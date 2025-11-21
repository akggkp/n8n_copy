# services/backtesting-service/app/main.py
# FastAPI service for strategy backtesting

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Backtesting Service",
    description="Service for backtesting trading strategies with performance metrics",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage (replace with database in production)
strategies_db = {}
backtests_db = {}
strategy_counter = 0
backtest_counter = 0


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RiskManagement(BaseModel):
    """Risk management parameters"""
    position_size: float = 0.1  # Fraction of capital per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit


class StrategyCreate(BaseModel):
    """Create strategy request"""
    name: str
    entry_rules: List[str]
    exit_rules: List[str]
    risk_management: RiskManagement


class StrategyResponse(BaseModel):
    """Strategy creation response"""
    strategy_id: int
    name: str
    created_at: datetime


class BacktestRequest(BaseModel):
    """Backtest execution request"""
    strategy_id: int
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 10000.0


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_return: float
    num_trades: int
    avg_trade_duration: float
    profit_factor: float


class BacktestResponse(BaseModel):
    """Backtest execution response"""
    backtest_id: int
    strategy_id: int
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    metrics: PerformanceMetrics
    timestamp: datetime


# ============================================================================
# MOCK DATA GENERATOR (Replace with real market data API)
# ============================================================================

def generate_mock_price_data(symbol: str, start_date: str, end_date: str, num_points: int = 365):
    """Generate mock OHLCV data for testing"""
    # Simple random walk for prices
    np.random.seed(42)
    
    base_price = 50000 if "BTC" in symbol else 100
    returns = np.random.normal(0.001, 0.02, num_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': i,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return data


def calculate_indicators(data: List[Dict]) -> List[Dict]:
    """Calculate technical indicators"""
    closes = np.array([d['close'] for d in data])
    
    # RSI (14-period)
    rsi = []
    for i in range(len(closes)):
        if i < 14:
            rsi.append(50)  # Neutral RSI for first 14 periods
        else:
            gains = []
            losses = []
            for j in range(i-14, i):
                diff = closes[j+1] - closes[j]
                if diff > 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(diff))
            
            avg_gain = np.mean(gains) if gains else 0.001
            avg_loss = np.mean(losses) if losses else 0.001
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
            rsi.append(rsi_val)
    
    # Add indicators to data
    for i, d in enumerate(data):
        d['rsi'] = rsi[i]
        d['sma_20'] = np.mean(closes[max(0, i-20):i+1])
        d['resistance'] = d['high'] * 1.02  # Mock resistance
        d['support'] = d['low'] * 0.98      # Mock support
    
    return data


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def run_backtest_simulation(
    strategy: Dict,
    price_data: List[Dict],
    initial_capital: float
) -> Dict:
    """
    Run backtest simulation
    
    Args:
        strategy: Strategy configuration
        price_data: OHLCV price data with indicators
        initial_capital: Starting capital
    
    Returns:
        Dict with trades, equity curve, and metrics
    """
    try:
        capital = initial_capital
        position = None  # {'entry_price', 'shares', 'stop_loss', 'take_profit'}
        trades = []
        equity_curve = [initial_capital]
        
        entry_rules = strategy['entry_rules']
        exit_rules = strategy['exit_rules']
        risk = strategy['risk_management']
        
        for i, bar in enumerate(price_data):
            if i == 0:
                continue
            
            # Check if we have a position
            if position is None:
                # Check entry conditions
                should_enter = evaluate_entry_rules(entry_rules, bar, price_data[:i])
                
                if should_enter:
                    # Enter position
                    position_size = capital * risk['position_size']
                    shares = position_size / bar['close']
                    
                    position = {
                        'entry_price': bar['close'],
                        'entry_time': i,
                        'shares': shares,
                        'stop_loss': bar['close'] * (1 - risk['stop_loss_pct']),
                        'take_profit': bar['close'] * (1 + risk['take_profit_pct'])
                    }
                    
                    capital -= position_size
                    logger.debug(f"Enter at {bar['close']:.2f}, shares={shares:.4f}")
            
            else:
                # Check exit conditions
                should_exit = (
                    bar['low'] <= position['stop_loss'] or  # Stop loss hit
                    bar['high'] >= position['take_profit'] or  # Take profit hit
                    evaluate_exit_rules(exit_rules, bar, price_data[:i])  # Exit rules
                )
                
                if should_exit:
                    # Exit position
                    exit_price = bar['close']
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    pnl_pct = (exit_price / position['entry_price'] - 1) * 100
                    
                    capital += position['shares'] * exit_price
                    
                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'duration': i - position['entry_time']
                    })
                    
                    logger.debug(f"Exit at {exit_price:.2f}, PnL={pnl:.2f} ({pnl_pct:.2f}%)")
                    position = None
            
            # Update equity curve
            current_equity = capital
            if position:
                current_equity += position['shares'] * bar['close']
            equity_curve.append(current_equity)
        
        # Calculate metrics
        metrics = calculate_performance_metrics(trades, equity_curve, initial_capital)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'final_capital': equity_curve[-1]
        }
    
    except Exception as e:
        logger.error(f"Error in backtest simulation: {str(e)}")
        raise


def evaluate_entry_rules(rules: List[str], bar: Dict, history: List[Dict]) -> bool:
    """Evaluate entry conditions"""
    try:
        for rule in rules:
            rule_lower = rule.lower()
            
            if 'rsi < 30' in rule_lower or 'rsi<30' in rule_lower:
                if bar.get('rsi', 50) >= 30:
                    return False
            
            elif 'macd' in rule_lower and 'histogram > 0' in rule_lower:
                # Mock MACD check
                if np.random.random() < 0.5:
                    return False
            
            elif 'close > resistance' in rule_lower or 'breakout' in rule_lower:
                if bar['close'] <= bar.get('resistance', bar['high']):
                    return False
            
            elif 'default_entry' in rule_lower:
                # Default entry: buy when RSI < 40
                if bar.get('rsi', 50) >= 40:
                    return False
        
        return True
    except Exception as e:
        logger.error(f"Error evaluating entry rules: {str(e)}")
        return False


def evaluate_exit_rules(rules: List[str], bar: Dict, history: List[Dict]) -> bool:
    """Evaluate exit conditions"""
    try:
        for rule in rules:
            rule_lower = rule.lower()
            
            if 'rsi > 70' in rule_lower or 'rsi>70' in rule_lower:
                if bar.get('rsi', 50) > 70:
                    return True
            
            elif 'default_exit' in rule_lower:
                # Default exit: sell when RSI > 60
                if bar.get('rsi', 50) > 60:
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error evaluating exit rules: {str(e)}")
        return False


def calculate_performance_metrics(trades: List[Dict], equity_curve: List[float], initial_capital: float) -> Dict:
    """Calculate performance metrics"""
    try:
        if not trades:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'num_trades': 0,
                'avg_trade_duration': 0.0,
                'profit_factor': 0.0
            }
        
        # Win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        
        # Total return
        final_capital = equity_curve[-1]
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Average trade duration (bars)
        avg_duration = np.mean([t['duration'] for t in trades])
        
        # Profit factor
        gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate': round(win_rate, 4),
            'max_drawdown': round(max_dd, 4),
            'total_return': round(total_return, 2),
            'num_trades': len(trades),
            'avg_trade_duration': round(avg_duration, 2),
            'profit_factor': round(profit_factor, 2)
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "backtesting-service",
        "strategies_count": len(strategies_db),
        "backtests_count": len(backtests_db)
    }


@app.post("/strategies", response_model=StrategyResponse, status_code=201)
async def create_strategy(strategy: StrategyCreate):
    """Create a new trading strategy"""
    try:
        global strategy_counter
        strategy_counter += 1
        
        strategy_data = {
            'id': strategy_counter,
            'name': strategy.name,
            'entry_rules': strategy.entry_rules,
            'exit_rules': strategy.exit_rules,
            'risk_management': strategy.risk_management.dict(),
            'created_at': datetime.utcnow()
        }
        
        strategies_db[strategy_counter] = strategy_data
        
        logger.info(f"Created strategy {strategy_counter}: {strategy.name}")
        
        return StrategyResponse(
            strategy_id=strategy_counter,
            name=strategy.name,
            created_at=strategy_data['created_at']
        )
    
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: int):
    """Get strategy by ID"""
    if strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return strategies_db[strategy_id]


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest for a strategy"""
    try:
        global backtest_counter
        
        # Get strategy
        if request.strategy_id not in strategies_db:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = strategies_db[request.strategy_id]
        
        # Generate price data
        price_data = generate_mock_price_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Add indicators
        price_data = calculate_indicators(price_data)
        
        # Run backtest
        result = run_backtest_simulation(
            strategy=strategy,
            price_data=price_data,
            initial_capital=request.initial_capital
        )
        
        # Store backtest result
        backtest_counter += 1
        backtest_data = {
            'id': backtest_counter,
            'strategy_id': request.strategy_id,
            'symbol': request.symbol,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'initial_capital': request.initial_capital,
            'final_capital': result['final_capital'],
            'trades': result['trades'],
            'equity_curve': result['equity_curve'],
            'metrics': result['metrics'],
            'timestamp': datetime.utcnow()
        }
        
        backtests_db[backtest_counter] = backtest_data
        
        logger.info(f"Backtest {backtest_counter} completed for strategy {request.strategy_id}")
        
        return BacktestResponse(
            backtest_id=backtest_counter,
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            final_capital=result['final_capital'],
            metrics=PerformanceMetrics(**result['metrics']),
            timestamp=backtest_data['timestamp']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/{backtest_id}/metrics")
async def get_backtest_metrics(backtest_id: int):
    """Get performance metrics for a backtest"""
    if backtest_id not in backtests_db:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    backtest = backtests_db[backtest_id]
    return backtest['metrics']


@app.get("/backtest/{backtest_id}/trades")
async def get_backtest_trades(backtest_id: int):
    """Get all trades from a backtest"""
    if backtest_id not in backtests_db:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    backtest = backtests_db[backtest_id]
    return {
        'backtest_id': backtest_id,
        'trades': backtest['trades'],
        'num_trades': len(backtest['trades'])
    }


@app.get("/backtest/{backtest_id}/equity")
async def get_equity_curve(backtest_id: int):
    """Get equity curve from a backtest"""
    if backtest_id not in backtests_db:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    backtest = backtests_db[backtest_id]
    return {
        'backtest_id': backtest_id,
        'equity_curve': backtest['equity_curve'],
        'initial_capital': backtest['initial_capital'],
        'final_capital': backtest['final_capital']
    }


@app.get("/")
async def root():
    """Service information"""
    return {
        "name": "Backtesting Service",
        "version": "1.0.0",
        "description": "Service for backtesting trading strategies with performance metrics",
        "docs_url": "/docs",
        "endpoints": {
            "create_strategy": "/strategies (POST)",
            "get_strategy": "/strategies/{strategy_id} (GET)",
            "run_backtest": "/backtest (POST)",
            "get_metrics": "/backtest/{backtest_id}/metrics (GET)",
            "get_trades": "/backtest/{backtest_id}/trades (GET)",
            "get_equity": "/backtest/{backtest_id}/equity (GET)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("BACKTEST_PORT", 8001)),
        workers=1
    )
