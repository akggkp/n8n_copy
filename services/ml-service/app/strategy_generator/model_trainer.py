"""Strategy Generator - Generate trading strategies from concepts"""

import logging
from typing import List, Dict, Any
import json
import os
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logger = logging.getLogger(__name__)

Base = declarative_base()

class MLStrategy(Base):
    __tablename__ = 'ml_strategies'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String, nullable=False, index=True)
    strategy_name = Column(String)
    concepts = Column(JSON)
    indicators = Column(JSON)
    patterns = Column(JSON)
    strategy_rules = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)

class StrategyGenerator:
    """Generate trading strategies from extracted concepts"""
    
    def __init__(self):
        DATABASE_URL = os.getenv("DATABASE_URL")
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def generate_strategies(self, video_id: str, concepts: List[str], 
                          indicators: List[str], patterns: List[str]) -> List[Dict[str, Any]]:
        """
        Generate strategies from concepts
        
        Args:
            video_id: Video identifier
            concepts: Trading concepts extracted
            indicators: Technical indicators
            patterns: Chart patterns
            
        Returns:
            List of generated strategies
        """
        strategies = []
        
        logger.info(f"Generating strategies for {video_id}")
        logger.info(f"Concepts: {concepts}")
        logger.info(f"Indicators: {indicators}")
        logger.info(f"Patterns: {patterns}")
        
        # Strategy 1: Mean Reversion
        if 'rsi' in [i.lower() for i in indicators]:
            strategy = {
                'name': 'RSI Mean Reversion',
                'indicators': ['RSI'],
                'entry': 'RSI < 30',
                'exit': 'RSI > 70',
                'risk': '2%',
                'reward': '5%',
                'concepts': concepts
            }
            strategies.append(strategy)
        
        # Strategy 2: MACD Momentum
        if 'macd' in [i.lower() for i in indicators]:
            strategy = {
                'name': 'MACD Momentum',
                'indicators': ['MACD'],
                'entry': 'MACD Crossover (bullish)',
                'exit': 'MACD Crossover (bearish)',
                'risk': '2%',
                'reward': '6%',
                'concepts': concepts
            }
            strategies.append(strategy)
        
        # Strategy 3: EMA Crossover
        if 'ema' in [i.lower() for i in indicators]:
            strategy = {
                'name': 'EMA Crossover',
                'indicators': ['EMA(9)', 'EMA(21)'],
                'entry': 'EMA9 > EMA21 (bullish)',
                'exit': 'EMA9 < EMA21 (bearish)',
                'risk': '2%',
                'reward': '5%',
                'concepts': concepts
            }
            strategies.append(strategy)
        
        # Strategy 4: Bollinger Bands
        if 'bollinger' in [i.lower() for i in indicators]:
            strategy = {
                'name': 'Bollinger Bands Reversion',
                'indicators': ['Bollinger Bands'],
                'entry': 'Price touches lower band',
                'exit': 'Price reaches middle band',
                'risk': '2%',
                'reward': '4%',
                'concepts': concepts
            }
            strategies.append(strategy)
        
        # Strategy 5: Support/Resistance
        if 'support' in [c.lower() for c in concepts] or 'resistance' in [c.lower() for c in concepts]:
            strategy = {
                'name': 'Support Resistance Breakout',
                'indicators': [],
                'entry': 'Breakout above resistance',
                'exit': 'Closes below support',
                'risk': '3%',
                'reward': '7%',
                'concepts': concepts,
                'patterns': patterns
            }
            strategies.append(strategy)
        
        # Save to database
        session = self.Session()
        try:
            for idx, strategy in enumerate(strategies):
                db_strategy = MLStrategy(
                    strategy_id=f"{video_id}_strategy_{idx}",
                    strategy_name=strategy['name'],
                    concepts=concepts,
                    indicators=indicators,
                    patterns=patterns,
                    strategy_rules=strategy
                )
                session.add(db_strategy)
            
            session.commit()
            logger.info(f"Saved {len(strategies)} strategies to database")
        
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving strategies: {str(e)}")
        finally:
            session.close()
        
        return strategies
