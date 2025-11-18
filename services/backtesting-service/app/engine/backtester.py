"""
Backtesting Engine - Strategy validation and performance analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple
import os
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String, nullable=False, index=True)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_return = Column(Float)
    total_pnl = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    strategy_data = Column(JSON)
    is_profitable = Column(Integer)
    tested_at = Column(DateTime, default=datetime.now)

class BacktestEngine:
    """Strategy backtesting engine"""
    
    def __init__(self):
        DATABASE_URL = os.getenv("DATABASE_URL")
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch historical OHLC data
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            # Placeholder: Load from database or API
            # For now, generate dummy data for testing
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Generate sample data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            data = pd.DataFrame({
                'datetime': dates,
                'open': np.random.uniform(100, 150, len(dates)),
                'high': np.random.uniform(150, 160, len(dates)),
                'low': np.random.uniform(90, 100, len(dates)),
                'close': np.random.uniform(100, 150, len(dates)),
                'volume': np.random.uniform(1000, 10000, len(dates))
            })
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_signals(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate entry and exit signals based on strategy
        
        Args:
            data: Historical OHLC data
            strategy: Strategy definition
            
        Returns:
            DataFrame with signals
        """
        data = data.copy()
        data['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        data['position'] = 0
        
        # Simple logic: for demonstration
        # In production, implement actual strategy logic
        
        return data
    
    def run_simulation(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest simulation
        
        Args:
            data: OHLC data with signals
            strategy: Strategy definition
            
        Returns:
            Performance metrics
        """
        if data.empty:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "total_pnl": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "avg_win": 0,
                "avg_loss": 0
            }
        
        # Calculate signals
        data = self.calculate_signals(data, strategy)
        
        # Simulate trades
        trades = []
        entry_price = None
        entry_date = None
        
        for idx, row in data.iterrows():
            if row['signal'] == 1 and entry_price is None:
                entry_price = row['close']
                entry_date = row['datetime']
            elif row['signal'] == -1 and entry_price is not None:
                exit_price = row['close']
                pnl = exit_price - entry_price
                trades.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': pnl,
                    'profitable': pnl > 0
                })
                entry_price = None
                entry_date = None
        
        # Calculate metrics
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "total_pnl": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "avg_win": 0,
                "avg_loss": 0
            }
        
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['profitable'] == True]
        losing_trades = trades_df[trades_df['profitable'] == False]
        
        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_pnl = trades_df['pnl'].sum()
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        # Calculate Sharpe Ratio (simplified)
        returns = trades_df['pnl'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe_ratio),
            "total_pnl": float(total_pnl),
            "total_return": float((total_pnl / 1000) * 100),  # Assuming 1000 initial
            "max_drawdown": 0,  # Simplified
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss)
        }
    
    def backtest_strategy(self, strategy_id: str, strategy_data: Dict[str, Any], 
                         symbol: str = "NIFTY", timeframe: str = "15m",
                         start_date: str = "2024-01-01", end_date: str = "2024-12-31") -> bool:
        """
        Run complete backtest for a strategy
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_data: Strategy definition
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            True if strategy is profitable
        """
        try:
            logger.info(f"Running backtest for {strategy_id}")
            
            # Fetch data
            data = self.fetch_historical_data(symbol, start_date, end_date, timeframe)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return False
            
            # Run simulation
            metrics = self.run_simulation(data, strategy_data)
            
            # Determine profitability
            min_wr = float(os.getenv("MIN_WIN_RATE_TO_SAVE", 55))
            min_pf = float(os.getenv("MIN_PROFIT_FACTOR", 1.5))
            min_sr = float(os.getenv("MIN_SHARPE_RATIO", 0.5))
            
            is_profitable = (
                metrics['win_rate'] > min_wr and
                metrics['profit_factor'] > min_pf and
                metrics['sharpe_ratio'] > min_sr
            )
            
            # Save to database
            session = self.Session()
            result = BacktestResult(
                strategy_id=strategy_id,
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                total_return=metrics['total_return'],
                total_pnl=metrics['total_pnl'],
                total_trades=metrics['total_trades'],
                winning_trades=metrics['winning_trades'],
                losing_trades=metrics['losing_trades'],
                avg_win=metrics['avg_win'],
                avg_loss=metrics['avg_loss'],
                strategy_data=strategy_data,
                is_profitable=1 if is_profitable else 0,
                tested_at=datetime.now()
            )
            session.add(result)
            session.commit()
            session.close()
            
            logger.info(f"Backtest completed for {strategy_id}")
            logger.info(f"Results: WR={metrics['win_rate']:.1f}%, PF={metrics['profit_factor']:.2f}, SR={metrics['sharpe_ratio']:.2f}")
            
            return is_profitable
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return False
