"""Celery tasks for video processing pipeline"""
import os
import sys
import requests
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from celery import chain, group
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from app.celery_app import celery_app
from app.models import db, ProcessedVideo, ProvenStrategy, MlStrategy, KeywordHit, Transcript, Embedding
from app.database import SessionLocal
from app.feature_engineering import FeatureEngineer
from app.backtest_client import BacktestClient

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Database connection for tasks
engine = create_engine(Config.DATABASE_URL)
TaskSession = sessionmaker(bind=engine)

@contextmanager
def db_session():
    """Provide a transactional scope around a series of operations."""
    session = TaskSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# ==================== Individual Tasks ====================

@celery_app.task(bind=True, name='app.tasks.validate_video')
def validate_video(self, video_id, file_path, filename):
    """
    Task 1: Validate video file and register in database
    """
    logger.info(f"[Task 1/7] Validating video: {video_id}")
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Video size: {file_size_mb:.2f} MB")
        
        with db_session() as session:
            existing = session.query(ProcessedVideo).filter_by(video_id=video_id).first()
            if existing:
                logger.warning(f"Video {video_id} already processed")
                return {'status': 'skipped', 'video_id': video_id, 'reason': 'Already processed'}
            
            new_video = ProcessedVideo(
                video_id=video_id,
                filename=filename,
                status='validating',
                created_at=datetime.now()
            )
            session.add(new_video)
        
        logger.info(f"✓ Video validated: {video_id}")
        return {
            'status': 'success',
            'video_id': video_id,
            'file_path': file_path,
            'filename': filename,
            'file_size_mb': file_size_mb
        }
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)

@celery_app.task(bind=True, name='app.tasks.process_video')
def process_video(self, video_data):
    """
    Task 2: Process video - extract frames, detect charts, transcribe
    """
    video_id = video_data['video_id']
    logger.info(f"[Task 2/7] Processing video: {video_id}")
    
    try:
        with db_session() as session:
            video = session.query(ProcessedVideo).filter_by(video_id=video_id).first()
            if video:
                video.status = 'processing'

        url = f"{Config.VIDEO_PROCESSOR_URL}/process"
        payload = {
            'video_id': video_id,
            'file_path': video_data['file_path'],
            'use_cascade': Config.USE_CASCADE,
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD
        }
        
        logger.info(f"Calling video processor: {url}")
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"✓ Video processed: {len(result.get('detected_charts', []))} charts detected")
        
        with db_session() as session:
            video = session.query(ProcessedVideo).filter_by(video_id=video_id).first()
            if video:
                video.transcription = result.get('transcription', '')
                video.detected_charts = result.get('detected_charts', [])
                video.processing_stats = result.get('processing_stats', {})
        
        return {**video_data, **result}
        
    except Exception as e:
        logger.error(f"✗ Processing failed: {str(e)}")
        with db_session() as session:
            video = session.query(ProcessedVideo).filter_by(video_id=video_id).first()
            if video:
                video.status = 'failed'
        raise self.retry(exc=e, countdown=120)

@celery_app.task(bind=True, name='app.tasks.generate_embeddings')
def generate_embeddings(self, data):
    """
    Task X/7: Generate embeddings for transcript segments
    """
    video_id = data['video_id']
    logger.info(f"[Task X/7] Generating embeddings for: {video_id}")
    
    try:
        # As per EMBEDDINGS_SERVICE_GUIDE.md, this task requires 'transcript_segments'.
        # It's assumed a prior task provides this in the data dictionary.
        if 'transcript_segments' not in data:
            raise ValueError("Input data for generate_embeddings must contain 'transcript_segments'")
            
        transcript_segments = data['transcript_segments']

        embeddings_url = os.getenv('EMBEDDINGS_SERVICE_URL', 'http://localhost:8004')
        
        response = requests.post(
            f"{embeddings_url}/embed",
            json={
                'media_item_id': video_id,
                'embedding_type': 'transcript',
                'segments': transcript_segments
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"✓ Embeddings generated: {result.get('embeddings_created', 0)}")
        
        # Pass through the results from this task
        return {**data, **result}
        
    except Exception as e:
        logger.error(f"✗ Embeddings generation failed for {video_id}: {str(e)}")
        with db_session() as session:
            video = session.query(ProcessedVideo).filter_by(video_id=video_id).first()
            if video:
                video.status = 'failed'
        raise self.retry(exc=e, countdown=120)


@celery_app.task(bind=True, name='app.tasks.extract_features')
def extract_features(self, previous_result):
    """Extract ML features from processing results"""
    try:
        # Imports are global now
        
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
        # Pass through previous_result to avoid breaking the chain on failure
        return {**previous_result, 'status': 'failed', 'error': str(e)}


@celery_app.task(bind=True, name='app.tasks.generate_strategy')
def generate_strategy(self, previous_result):
    """Generate trading strategy from features"""
    try:
        # Imports are global now
        
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
                **previous_result,
                'status': 'skipped',
                'reason': 'Backtest service unavailable',
            }
        
        strategy_name = f"Video_Strategy_{media_item_id}"
        
        strategy_response = backtest_client.create_strategy(
            strategy_name=strategy_name,
            entry_rules=entry_rules or ["default_entry"],
            exit_rules=exit_rules or ["default_exit"],
            risk_params=risk_params
        )
        
        if not strategy_response:
            return {**previous_result, 'status': 'failed', 'error': 'Strategy creation failed'}
        
        strategy_id = strategy_response.get('strategy_id')
        
        logger.info(f"Created strategy {strategy_id} for media_item {media_item_id}")
        
        return {
            **previous_result,
            'status': 'success',
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'entry_rules': entry_rules,
            'exit_rules': exit_rules,
            'risk_params': risk_params
        }
    
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        return {**previous_result, 'status': 'failed', 'error': str(e)}


@celery_app.task(bind=True, name='app.tasks.backtest_strategy')
def backtest_strategy(self, previous_result):
    """Run backtest for generated strategy"""
    try:
        # Imports are global now
        
        strategy_id = previous_result.get('strategy_id')
        media_item_id = previous_result.get('media_item_id')
        
        if not strategy_id:
            return {**previous_result, 'status': 'skipped', 'reason': 'No strategy_id provided'}
        
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
            return {**previous_result, 'status': 'failed', 'error': 'Backtest execution failed'}
        
        backtest_id = backtest_result.get('backtest_id')
        
        # Get performance metrics
        metrics = backtest_client.get_performance_metrics(backtest_id)
        
        if not metrics:
            return {**previous_result, 'status': 'failed', 'error': 'Failed to get performance metrics'}
        
        logger.info(f"Backtest complete for strategy {strategy_id}: {metrics}")
        
        return {
            **previous_result,
            'status': 'success',
            'backtest_id': backtest_id,
            'metrics': metrics
        }
    
    except Exception as e:
        logger.error(f"Error backtesting strategy: {str(e)}")
        return {**previous_result, 'status': 'failed', 'error': str(e)}


@celery_app.task(bind=True, name='app.tasks.evaluate_and_promote')
def evaluate_and_promote(self, previous_result):
    """Evaluate strategy and promote if it meets thresholds"""
    try:
        # Imports are global now
        
        metrics = previous_result.get('metrics', {})
        strategy_id = previous_result.get('strategy_id')
        media_item_id = previous_result.get('media_item_id')
        
        if not metrics:
            return {**previous_result, 'status': 'skipped', 'reason': 'No metrics provided'}
        
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
                    **previous_result,
                    'status': 'promoted',
                    'proven_strategy_id': proven_strategy.id
                }
            
            finally:
                db.close()
        else:
            logger.info(f"Strategy {strategy_id} did NOT meet promotion thresholds")
            
            return {
                **previous_result,
                'status': 'not_promoted',
                'reason': 'Failed performance thresholds'
            }
    
    except Exception as e:
        logger.error(f"Error evaluating/promoting strategy: {str(e)}")
        return {**previous_result, 'status': 'failed', 'error': str(e)}


@celery_app.task(bind=True, name='app.tasks.run_full_pipeline')
def run_full_pipeline(self, media_item_id, file_path, filename):
    """Run complete video processing + strategy generation pipeline"""
    try:
        logger.info(f"Starting full pipeline for media_item {media_item_id}")
        
        # Build pipeline chain
        pipeline = chain(
            validate_video.s(media_item_id, file_path, filename),
            process_video.s(),
            detect_keywords.s(),
            generate_clips.s(),
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
