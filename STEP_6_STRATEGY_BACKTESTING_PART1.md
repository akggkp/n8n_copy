# STEP_6_STRATEGY_BACKTESTING_FRAMEWORK.md
# Step 6: Strategy & Backtesting Framework - Complete Implementation Guide

## Overview

Step 6 builds a complete strategy generation and validation framework that:
1. Extracts features from keyword hits, transcripts, and embeddings
2. Uses ML/RL agents to generate trade signals
3. Backtests strategies with realistic constraints
4. Validates performance metrics (Sharpe ratio, win rate, drawdown)
5. Promotes proven strategies to production

---

## Architecture

```
Strategy & Backtesting Pipeline
├── Feature Engineering Module
│   ├── Extract keyword patterns
│   ├── Build context windows (±10s)
│   ├── Generate technical indicators from metadata
│   └── Create embedding-based similarity scores
├── ML/RL Agent Module
│   ├── Train on historical features
│   ├── Generate entry/exit signals
│   └── Optimize hyperparameters
├── Backtesting Service (Port 8001)
│   ├── Simulate trades with slippage/fees
│   ├── Calculate performance metrics
│   └── Generate equity curves
└── Strategy Promotion Logic
    ├── Validate against thresholds
    ├── Store proven strategies
    └── Enable for live trading
```

---

## Feature Engineering Module

Create `orchestrator/app/feature_engineering.py`:

```python
# orchestrator/app/feature_engineering.py
# Extract features from processed videos for ML/RL training

import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract trading strategy features from video processing results"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_keyword_features(self, keyword_hits: List[Dict]) -> Dict:
        """
        Extract features from keyword detection results
        
        Args:
            keyword_hits: List of keyword hit dicts with timestamp, confidence, category
        
        Returns:
            Feature dict with counts, frequencies, confidence stats
        """
        try:
            if not keyword_hits:
                return self._empty_keyword_features()
            
            features = {}
            
            # Category counts
            categories = [hit.get('category', 'unknown') for hit in keyword_hits]
            features['technical_indicator_count'] = categories.count('technical_indicator')
            features['price_action_count'] = categories.count('price_action')
            features['risk_management_count'] = categories.count('risk_management')
            features['order_type_count'] = categories.count('order_type')
            
            # Keyword diversity
            unique_keywords = len(set([hit.get('keyword', '') for hit in keyword_hits]))
            features['keyword_diversity'] = unique_keywords / len(keyword_hits) if keyword_hits else 0.0
            
            # Confidence statistics
            confidences = [hit.get('confidence', 0.0) for hit in keyword_hits]
            features['avg_confidence'] = np.mean(confidences) if confidences else 0.0
            features['min_confidence'] = np.min(confidences) if confidences else 0.0
            features['max_confidence'] = np.max(confidences) if confidences else 0.0
            
            # Temporal features
            timestamps = sorted([hit.get('start_time', 0.0) for hit in keyword_hits])
            if len(timestamps) > 1:
                features['keyword_frequency'] = len(timestamps) / (timestamps[-1] - timestamps[0] + 1)
                features['avg_time_between_keywords'] = np.mean(np.diff(timestamps))
            else:
                features['keyword_frequency'] = 0.0
                features['avg_time_between_keywords'] = 0.0
            
            # Binary flags for important concepts
            keywords_lower = [hit.get('keyword', '').lower() for hit in keyword_hits]
            features['has_rsi'] = 1 if 'rsi' in keywords_lower else 0
            features['has_macd'] = 1 if 'macd' in keywords_lower else 0
            features['has_support_resistance'] = 1 if any(k in keywords_lower for k in ['support', 'resistance']) else 0
            features['has_breakout'] = 1 if 'breakout' in keywords_lower else 0
            features['has_stop_loss'] = 1 if 'stop loss' in keywords_lower or 'stop-loss' in keywords_lower else 0
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting keyword features: {str(e)}")
            return self._empty_keyword_features()
    
    def extract_transcript_features(self, transcripts: List[Dict]) -> Dict:
        """
        Extract features from transcript analysis
        
        Args:
            transcripts: List of transcript segment dicts with text, timestamps
        
        Returns:
            Feature dict with text statistics
        """
        try:
            if not transcripts:
                return self._empty_transcript_features()
            
            features = {}
            
            # Text length statistics
            texts = [t.get('text', '') for t in transcripts]
            text_lengths = [len(text.split()) for text in texts]
            
            features['total_words'] = sum(text_lengths)
            features['avg_words_per_segment'] = np.mean(text_lengths) if text_lengths else 0.0
            features['total_segments'] = len(transcripts)
            
            # Speaking rate (words per minute)
            if transcripts and len(transcripts) > 1:
                duration = transcripts[-1].get('end_time', 0) - transcripts[0].get('start_time', 0)
                features['speaking_rate_wpm'] = (features['total_words'] / duration) * 60 if duration > 0 else 0.0
            else:
                features['speaking_rate_wpm'] = 0.0
            
            # Educational signal detection (simple heuristics)
            full_text = " ".join(texts).lower()
            
            features['has_explanation'] = 1 if any(word in full_text for word in ['because', 'therefore', 'reason', 'due to']) else 0
            features['has_examples'] = 1 if any(word in full_text for word in ['example', 'for instance', 'such as']) else 0
            features['has_steps'] = 1 if any(word in full_text for word in ['first', 'second', 'third', 'step']) else 0
            features['question_density'] = full_text.count('?') / len(texts) if texts else 0.0
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting transcript features: {str(e)}")
            return self._empty_transcript_features()
    
    def extract_embedding_features(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Extract features from embedding vectors
        
        Args:
            embeddings: List of embedding vectors (numpy arrays)
        
        Returns:
            Feature dict with embedding statistics
        """
        try:
            if not embeddings or len(embeddings) == 0:
                return self._empty_embedding_features()
            
            features = {}
            
            # Stack embeddings
            emb_matrix = np.vstack(embeddings)
            
            # Embedding statistics
            features['embedding_dim'] = emb_matrix.shape[1]
            features['avg_embedding_norm'] = np.mean(np.linalg.norm(emb_matrix, axis=1))
            
            # Diversity: average pairwise cosine similarity
            if len(embeddings) > 1:
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)
                
                features['avg_embedding_similarity'] = np.mean(similarities) if similarities else 0.0
                features['embedding_diversity'] = 1.0 - features['avg_embedding_similarity']
            else:
                features['avg_embedding_similarity'] = 0.0
                features['embedding_diversity'] = 0.0
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting embedding features: {str(e)}")
            return self._empty_embedding_features()
    
    def build_feature_vector(
        self,
        keyword_hits: List[Dict],
        transcripts: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Build complete feature vector for a media item
        
        Args:
            keyword_hits: Keyword detection results
            transcripts: Transcript segments
            embeddings: Optional embedding vectors
        
        Returns:
            Complete feature dict
        """
        try:
            features = {}
            
            # Extract features from each source
            keyword_features = self.extract_keyword_features(keyword_hits)
            transcript_features = self.extract_transcript_features(transcripts)
            
            if embeddings:
                embedding_features = self.extract_embedding_features(embeddings)
            else:
                embedding_features = self._empty_embedding_features()
            
            # Combine all features
            features.update(keyword_features)
            features.update(transcript_features)
            features.update(embedding_features)
            
            # Add metadata
            features['timestamp'] = datetime.utcnow().isoformat()
            features['feature_count'] = len(features)
            
            # Store feature names for later use
            self.feature_names = list(features.keys())
            
            logger.info(f"Built feature vector with {len(features)} features")
            return features
        
        except Exception as e:
            logger.error(f"Error building feature vector: {str(e)}")
            return {}
    
    def features_to_array(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array for ML models"""
        try:
            # Exclude non-numeric fields
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            return np.array(list(numeric_features.values()))
        except Exception as e:
            logger.error(f"Error converting features to array: {str(e)}")
            return np.array([])
    
    def _empty_keyword_features(self) -> Dict:
        """Return empty keyword feature dict"""
        return {
            'technical_indicator_count': 0,
            'price_action_count': 0,
            'risk_management_count': 0,
            'order_type_count': 0,
            'keyword_diversity': 0.0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'keyword_frequency': 0.0,
            'avg_time_between_keywords': 0.0,
            'has_rsi': 0,
            'has_macd': 0,
            'has_support_resistance': 0,
            'has_breakout': 0,
            'has_stop_loss': 0
        }
    
    def _empty_transcript_features(self) -> Dict:
        """Return empty transcript feature dict"""
        return {
            'total_words': 0,
            'avg_words_per_segment': 0.0,
            'total_segments': 0,
            'speaking_rate_wpm': 0.0,
            'has_explanation': 0,
            'has_examples': 0,
            'has_steps': 0,
            'question_density': 0.0
        }
    
    def _empty_embedding_features(self) -> Dict:
        """Return empty embedding feature dict"""
        return {
            'embedding_dim': 0,
            'avg_embedding_norm': 0.0,
            'avg_embedding_similarity': 0.0,
            'embedding_diversity': 0.0
        }
```

---

## Backtesting Service Client

Create `orchestrator/app/backtest_client.py`:

```python
# orchestrator/app/backtest_client.py
# Client for interacting with backtesting service

import requests
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class BacktestClient:
    """Client for backtesting trading strategies"""
    
    def __init__(self, backtest_url: str = "http://localhost:8001"):
        """
        Initialize backtesting client
        
        Args:
            backtest_url: Base URL for backtesting service
        """
        self.backtest_url = backtest_url
    
    def create_strategy(
        self,
        strategy_name: str,
        entry_rules: List[str],
        exit_rules: List[str],
        risk_params: Dict
    ) -> Optional[Dict]:
        """
        Create a new strategy configuration
        
        Args:
            strategy_name: Name of strategy
            entry_rules: List of entry condition strings
            exit_rules: List of exit condition strings
            risk_params: Dict with position_size, stop_loss, take_profit
        
        Returns:
            Strategy creation response with strategy_id
        """
        try:
            response = requests.post(
                f"{self.backtest_url}/strategies",
                json={
                    "name": strategy_name,
                    "entry_rules": entry_rules,
                    "exit_rules": exit_rules,
                    "risk_management": risk_params
                },
                timeout=30
            )
            
            if response.status_code != 201:
                logger.error(f"Strategy creation failed: {response.text}")
                return None
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            return None
    
    def run_backtest(
        self,
        strategy_id: int,
        symbol: str = "BTCUSDT",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 10000.0
    ) -> Optional[Dict]:
        """
        Run backtest for a strategy
        
        Args:
            strategy_id: ID of strategy to backtest
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital in USD
        
        Returns:
            Backtest results with performance metrics
        """
        try:
            response = requests.post(
                f"{self.backtest_url}/backtest",
                json={
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_capital": initial_capital
                },
                timeout=300  # 5 min timeout for backtest
            )
            
            if response.status_code != 200:
                logger.error(f"Backtest failed: {response.text}")
                return None
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return None
    
    def get_performance_metrics(self, backtest_id: int) -> Optional[Dict]:
        """
        Get detailed performance metrics for a backtest
        
        Args:
            backtest_id: ID of backtest run
        
        Returns:
            Performance metrics dict
        """
        try:
            response = requests.get(
                f"{self.backtest_url}/backtest/{backtest_id}/metrics",
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get metrics: {response.text}")
                return None
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return None
    
    def validate_strategy(
        self,
        performance_metrics: Dict,
        min_sharpe_ratio: float = 1.0,
        min_win_rate: float = 0.55,
        max_drawdown: float = 0.25
    ) -> bool:
        """
        Validate if strategy meets performance thresholds
        
        Args:
            performance_metrics: Dict with sharpe_ratio, win_rate, max_drawdown
            min_sharpe_ratio: Minimum Sharpe ratio (default: 1.0)
            min_win_rate: Minimum win rate (default: 55%)
            max_drawdown: Maximum acceptable drawdown (default: 25%)
        
        Returns:
            True if strategy passes all thresholds
        """
        try:
            sharpe = performance_metrics.get('sharpe_ratio', 0.0)
            win_rate = performance_metrics.get('win_rate', 0.0)
            drawdown = abs(performance_metrics.get('max_drawdown', 1.0))
            
            passed = (
                sharpe >= min_sharpe_ratio and
                win_rate >= min_win_rate and
                drawdown <= max_drawdown
            )
            
            if passed:
                logger.info(f"Strategy validation PASSED: Sharpe={sharpe:.2f}, WinRate={win_rate:.2%}, Drawdown={drawdown:.2%}")
            else:
                logger.warning(f"Strategy validation FAILED: Sharpe={sharpe:.2f}, WinRate={win_rate:.2%}, Drawdown={drawdown:.2%}")
            
            return passed
        
        except Exception as e:
            logger.error(f"Error validating strategy: {str(e)}")
            return False
    
    def health_check(self) -> bool:
        """Check if backtesting service is available"""
        try:
            response = requests.get(f"{self.backtest_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Backtest service health check failed: {str(e)}")
            return False
```

---

## Update Orchestrator Tasks

Add to `orchestrator/app/tasks.py`:

```python
@shared_task(bind=True, name='app.tasks.extract_features')
def extract_features(self, previous_result):
    """Extract ML features from processing results"""
    try:
        from app.feature_engineering import FeatureEngineer
        from app.models import KeywordHit, Transcript, Embedding
        from app.database import SessionLocal
        
        media_item_id = previous_result.get('media_item_id')
        
        db = SessionLocal()
        try:
            # Get keyword hits
            keyword_hits = db.query(KeywordHit).filter_by(media_item_id=media_item_id).all()
            keyword_hits_dict = [
                {
                    'keyword': kh.keyword,
                    'category': kh.category,
                    'start_time': kh.start_time,
                    'end_time': kh.end_time,
                    'confidence': kh.confidence
                }
                for kh in keyword_hits
            ]
            
            # Get transcripts
            transcripts = db.query(Transcript).filter_by(media_item_id=media_item_id).all()
            transcripts_dict = [
                {
                    'text': t.text,
                    'start_time': t.start_time,
                    'end_time': t.end_time
                }
                for t in transcripts
            ]
            
            # Get embeddings (optional)
            embeddings = db.query(Embedding).filter_by(media_item_id=media_item_id).all()
            embedding_vectors = [e.embedding_vector for e in embeddings] if embeddings else None
            
            # Extract features
            feature_engineer = FeatureEngineer()
            features = feature_engineer.build_feature_vector(
                keyword_hits=keyword_hits_dict,
                transcripts=transcripts_dict,
                embeddings=embedding_vectors
            )
            
            logger.info(f"Extracted {len(features)} features for media_item {media_item_id}")
            
            return {
                'status': 'success',
                'media_item_id': media_item_id,
                'features': features,
                'feature_count': len(features)
            }
        
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return {'status': 'failed', 'error': str(e)}


@shared_task(bind=True, name='app.tasks.generate_strategy')
def generate_strategy(self, previous_result):
    """Generate trading strategy from features"""
    try:
        import os
        from app.backtest_client import BacktestClient
        
        features = previous_result.get('features', {})
        media_item_id = previous_result.get('media_item_id')
        
        # Build strategy rules from features
        entry_rules = []
        exit_rules = []
        
        # Entry rules based on detected concepts
        if features.get('has_rsi', 0) == 1:
            entry_rules.append("RSI < 30")  # Oversold
        
        if features.get('has_macd', 0) == 1:
            entry_rules.append("MACD_histogram > 0")  # Bullish crossover
        
        if features.get('has_breakout', 0) == 1:
            entry_rules.append("close > resistance")  # Breakout
        
        # Exit rules based on risk management
        if features.get('has_stop_loss', 0) == 1:
            exit_rules.append("stop_loss_hit")
        
        exit_rules.append("take_profit_hit")
        exit_rules.append("RSI > 70")  # Overbought exit
        
        # Risk parameters
        risk_params = {
            "position_size": 0.1,  # 10% of capital per trade
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.06  # 6% take profit (3:1 R:R)
        }
        
        # Create strategy via backtest service
        backtest_url = os.getenv('BACKTEST_SERVICE_URL', 'http://localhost:8001')
        backtest_client = BacktestClient(backtest_url=backtest_url)
        
        if not backtest_client.health_check():
            logger.warning("Backtest service unavailable, skipping strategy creation")
            return {
                'status': 'skipped',
                'reason': 'Backtest service unavailable',
                'media_item_id': media_item_id
            }
        
        strategy_name = f"Video_Strategy_{media_item_id}"
        
        strategy_response = backtest_client.create_strategy(
            strategy_name=strategy_name,
            entry_rules=entry_rules or ["default_entry"],
            exit_rules=exit_rules or ["default_exit"],
            risk_params=risk_params
        )
        
        if not strategy_response:
            return {'status': 'failed', 'error': 'Strategy creation failed'}
        
        strategy_id = strategy_response.get('strategy_id')
        
        logger.info(f"Created strategy {strategy_id} for media_item {media_item_id}")
        
        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'entry_rules': entry_rules,
            'exit_rules': exit_rules,
            'risk_params': risk_params
        }
    
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        return {'status': 'failed', 'error': str(e)}


@shared_task(bind=True, name='app.tasks.backtest_strategy')
def backtest_strategy(self, previous_result):
    """Run backtest for generated strategy"""
    try:
        import os
        from app.backtest_client import BacktestClient
        
        strategy_id = previous_result.get('strategy_id')
        media_item_id = previous_result.get('media_item_id')
        
        if not strategy_id:
            return {'status': 'skipped', 'reason': 'No strategy_id provided'}
        
        # Run backtest
        backtest_url = os.getenv('BACKTEST_SERVICE_URL', 'http://localhost:8001')
        backtest_client = BacktestClient(backtest_url=backtest_url)
        
        backtest_result = backtest_client.run_backtest(
            strategy_id=strategy_id,
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0
        )
        
        if not backtest_result:
            return {'status': 'failed', 'error': 'Backtest execution failed'}
        
        backtest_id = backtest_result.get('backtest_id')
        
        # Get performance metrics
        metrics = backtest_client.get_performance_metrics(backtest_id)
        
        if not metrics:
            return {'status': 'failed', 'error': 'Failed to get performance metrics'}
        
        logger.info(f"Backtest complete for strategy {strategy_id}: {metrics}")
        
        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'strategy_id': strategy_id,
            'backtest_id': backtest_id,
            'metrics': metrics
        }
    
    except Exception as e:
        logger.error(f"Error backtesting strategy: {str(e)}")
        return {'status': 'failed', 'error': str(e)}


@shared_task(bind=True, name='app.tasks.evaluate_and_promote')
def evaluate_and_promote(self, previous_result):
    """Evaluate strategy and promote if it meets thresholds"""
    try:
        import os
        from app.backtest_client import BacktestClient
        from app.models import ProvenStrategy
        from app.database import SessionLocal
        
        metrics = previous_result.get('metrics', {})
        strategy_id = previous_result.get('strategy_id')
        media_item_id = previous_result.get('media_item_id')
        
        if not metrics:
            return {'status': 'skipped', 'reason': 'No metrics provided'}
        
        # Validate strategy
        backtest_url = os.getenv('BACKTEST_SERVICE_URL', 'http://localhost:8001')
        backtest_client = BacktestClient(backtest_url=backtest_url)
        
        min_sharpe = float(os.getenv('MIN_SHARPE_RATIO', 1.0))
        min_win_rate = float(os.getenv('MIN_WIN_RATE_PERCENT', 55)) / 100.0
        max_drawdown = float(os.getenv('MAX_DRAWDOWN_PERCENT', 25)) / 100.0
        
        is_valid = backtest_client.validate_strategy(
            performance_metrics=metrics,
            min_sharpe_ratio=min_sharpe,
            min_win_rate=min_win_rate,
            max_drawdown=max_drawdown
        )
        
        if is_valid:
            # Promote to proven strategies table
            db = SessionLocal()
            try:
                proven_strategy = ProvenStrategy(
                    media_item_id=media_item_id,
                    strategy_id=strategy_id,
                    sharpe_ratio=metrics.get('sharpe_ratio'),
                    win_rate=metrics.get('win_rate'),
                    max_drawdown=metrics.get('max_drawdown'),
                    total_return=metrics.get('total_return'),
                    num_trades=metrics.get('num_trades'),
                    avg_trade_duration=metrics.get('avg_trade_duration'),
                    status='promoted'
                )
                
                db.add(proven_strategy)
                db.commit()
                
                logger.info(f"Strategy {strategy_id} PROMOTED to proven_strategies")
                
                return {
                    'status': 'promoted',
                    'media_item_id': media_item_id,
                    'strategy_id': strategy_id,
                    'metrics': metrics,
                    'proven_strategy_id': proven_strategy.id
                }
            
            finally:
                db.close()
        else:
            logger.info(f"Strategy {strategy_id} did NOT meet promotion thresholds")
            
            return {
                'status': 'not_promoted',
                'media_item_id': media_item_id,
                'strategy_id': strategy_id,
                'metrics': metrics,
                'reason': 'Failed performance thresholds'
            }
    
    except Exception as e:
        logger.error(f"Error evaluating/promoting strategy: {str(e)}")
        return {'status': 'failed', 'error': str(e)}
```

---

## Update Pipeline Chain

Modify `run_full_pipeline` in `orchestrator/app/tasks.py`:

```python
@shared_task(bind=True, name='app.tasks.run_full_pipeline')
def run_full_pipeline(self, media_item_id, file_path, filename):
    """Run complete video processing + strategy generation pipeline"""
    try:
        logger.info(f"Starting full pipeline for media_item {media_item_id}")
        
        # Build pipeline chain
        pipeline = chain(
            validate_video.s(media_item_id, file_path, filename),
            process_video.s(file_path),
            detect_keywords.s(),
            generate_clips.s(file_path),
            extract_concepts.s(),
            generate_embeddings.s(),
            extract_features.s(),           # NEW: Feature engineering
            generate_strategy.s(),          # NEW: Strategy generation
            backtest_strategy.s(),          # NEW: Backtesting
            evaluate_and_promote.s()        # NEW: Validation & promotion
        )
        
        # Execute pipeline asynchronously
        result = pipeline.apply_async()
        
        return {
            'status': 'pipeline_started',
            'media_item_id': media_item_id,
            'task_id': result.id
        }
    
    except Exception as e:
        logger.error(f"Error starting pipeline: {str(e)}")
        return {'status': 'failed', 'error': str(e)}
```

---

## Database Models Update

Add to `orchestrator/app/models.py` (or create if not exists):

```python
class ProvenStrategy(Base):
    __tablename__ = "proven_strategies"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    strategy_id = Column(Integer)  # From backtesting service
    
    # Performance metrics
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    max_drawdown = Column(Float)
    total_return = Column(Float)
    num_trades = Column(Integer)
    avg_trade_duration = Column(Float)  # hours
    
    # Status
    status = Column(String(50), default='promoted')  # promoted, active, retired
    promoted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    media_item = relationship("MediaItem", backref="proven_strategies")
```

---

## Continue in next file for backtesting service implementation, testing, and deployment...

Would you like me to continue with:
1. Backtesting Service FastAPI implementation (Port 8001)
2. Docker configuration and requirements
3. Testing scripts and examples
4. Performance monitoring dashboard
5. Complete integration testing workflow