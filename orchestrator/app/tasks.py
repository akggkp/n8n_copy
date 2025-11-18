"""Celery tasks for video processing pipeline"""
import os
import sys
import requests
import logging
from datetime import datetime, timedelta
from celery import chain, group
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from app.celery_app import celery_app

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Database connection
engine = create_engine(Config.DATABASE_URL)
Session = sessionmaker(bind=engine)

# ==================== Individual Tasks ====================

@celery_app.task(bind=True, name='app.tasks.validate_video')
def validate_video(self, video_id, file_path, filename):
    """
    Task 1: Validate video file and register in database
    """
    logger.info(f"[Task 1/7] Validating video: {video_id}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # Get file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Video size: {file_size_mb:.2f} MB")
        
        # Check if already processed
        session = Session()
        existing = session.execute(
            text("SELECT id FROM processed_videos WHERE video_id = :vid"),
            {"vid": video_id}
        ).fetchone()
        
        if existing:
            logger.warning(f"Video {video_id} already processed")
            session.close()
            return {
                'status': 'skipped',
                'video_id': video_id,
                'reason': 'Already processed'
            }
        
        # Insert into database
        session.execute(
            text("""
                INSERT INTO processed_videos (video_id, filename, status, created_at)
                VALUES (:vid, :fname, :status, :created)
            """),
            {
                'vid': video_id,
                'fname': filename,
                'status': 'validating',
                'created': datetime.now()
            }
        )
        session.commit()
        session.close()
        
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
    logger.info(f"[Task 2/7] Processing video: {video_data['video_id']}")
    
    try:
        # Update status
        session = Session()
        session.execute(
            text("UPDATE processed_videos SET status = :status WHERE video_id = :vid"),
            {'status': 'processing', 'vid': video_data['video_id']}
        )
        session.commit()
        session.close()
        
        # Call video processor service
        url = f"{Config.VIDEO_PROCESSOR_URL}/process"
        payload = {
            'video_id': video_data['video_id'],
            'file_path': video_data['file_path'],
            'use_cascade': Config.USE_CASCADE,
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD
        }
        
        logger.info(f"Calling video processor: {url}")
        response = requests.post(url, json=payload, timeout=600)
        
        if response.status_code != 200:
            raise Exception(f"Video processor failed: {response.text}")
        
        result = response.json()
        logger.info(f"✓ Video processed: {result.get('detected_charts', 0)} charts detected")
        
        return {
            'status': 'success',
            'video_id': video_data['video_id'],
            'transcription': result.get('transcription', ''),
            'detected_charts': result.get('detected_charts', []),
            'processing_stats': result.get('processing_stats', {})
        }
        
    except Exception as e:
        logger.error(f"✗ Processing failed: {str(e)}")
        # Update status to failed
        session = Session()
        session.execute(
            text("UPDATE processed_videos SET status = :status WHERE video_id = :vid"),
            {'status': 'failed', 'vid': video_data['video_id']}
        )
        session.commit()
        session.close()
        raise self.retry(exc=e, countdown=120)


@celery_app.task(bind=True, name='app.tasks.extract_concepts')
def extract_concepts(self, video_result):
    """
    Task 3: Extract trading concepts from transcription and charts
    """
    logger.info(f"[Task 3/7] Extracting concepts: {video_result['video_id']}")
    
    try:
        # Call ML service
        url = f"{Config.ML_SERVICE_URL}/extract-concepts"
        payload = {
            'video_id': video_result['video_id'],
            'transcription': video_result['transcription'],
            'detected_charts': video_result['detected_charts']
        }
        
        logger.info(f"Calling ML service: {url}")
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code != 200:
            raise Exception(f"Concept extraction failed: {response.text}")
        
        result = response.json()
        concepts = result.get('concepts', [])
        
        logger.info(f"✓ Extracted {len(concepts)} trading concepts")
        
        return {
            'status': 'success',
            'video_id': video_result['video_id'],
            'transcription': video_result['transcription'],
            'detected_charts': video_result['detected_charts'],
            'concepts': concepts,
            'indicators': result.get('indicators', []),
            'patterns': result.get('patterns', [])
        }
        
    except Exception as e:
        logger.error(f"✗ Concept extraction failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, name='app.tasks.generate_strategy')
def generate_strategy(self, concept_data):
    """
    Task 4: Generate trading strategy using LLaMA
    """
    logger.info(f"[Task 4/7] Generating strategy: {concept_data['video_id']}")
    
    try:
        # Build prompt for LLaMA
        prompt = f"""
You are a trading strategy analyst. Based on the following trading concepts extracted from a video, 
generate a detailed trading strategy.

Transcription Summary:
{concept_data['transcription'][:500]}...

Detected Concepts: {', '.join(concept_data.get('concepts', []))}
Indicators: {', '.join(concept_data.get('indicators', []))}
Patterns: {', '.join(concept_data.get('patterns', []))}

Provide:
1. Strategy Name
2. Entry Rules (specific conditions)
3. Exit Rules (profit target and stop loss)
4. Risk Management (position sizing, max risk)
5. Indicators to use

Format your response as JSON with keys: strategy_name, entry_rules, exit_rules, risk_management, indicators
"""
        
        # Call Ollama
        url = f"{Config.OLLAMA_URL}/api/generate"
        payload = {
            'model': Config.OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False,
            'format': 'json',
            'options': {
                'temperature': Config.LLM_TEMPERATURE
            }
        }
        
        logger.info(f"Calling Ollama: {url}")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"LLaMA generation failed: {response.text}")
        
        result = response.json()
        strategy_text = result.get('response', '{}')
        
        # Parse JSON response
        import json
        try:
            strategy = json.loads(strategy_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            strategy = {
                'strategy_name': 'Generated Strategy',
                'entry_rules': strategy_text[:200],
                'exit_rules': 'Standard exit rules',
                'risk_management': '1-2% risk per trade',
                'indicators': concept_data.get('indicators', [])
            }
        
        logger.info(f"✓ Strategy generated: {strategy.get('strategy_name')}")
        
        return {
            'status': 'success',
            'video_id': concept_data['video_id'],
            'strategy': strategy,
            'concepts': concept_data.get('concepts', []),
            'indicators': concept_data.get('indicators', [])
        }
        
    except Exception as e:
        logger.error(f"✗ Strategy generation failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, name='app.tasks.backtest_strategy')
def backtest_strategy(self, strategy_data):
    """
    Task 5: Backtest the generated strategy
    """
    logger.info(f"[Task 5/7] Backtesting strategy: {strategy_data['video_id']}")
    
    try:
        # Call backtesting service
        url = f"{Config.BACKTESTING_SERVICE_URL}/backtest"
        payload = {
            'video_id': strategy_data['video_id'],
            'strategy': strategy_data['strategy'],
            'symbol': 'NIFTY',
            'timeframe': '15m',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
        
        logger.info(f"Calling backtesting service: {url}")
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code != 200:
            raise Exception(f"Backtesting failed: {response.text}")
        
        result = response.json()
        backtest_results = result.get('backtest_results', {})
        
        logger.info(f"✓ Backtest complete: Win Rate {backtest_results.get('win_rate', 0)}%")
        
        return {
            'status': 'success',
            'video_id': strategy_data['video_id'],
            'strategy': strategy_data['strategy'],
            'backtest_results': backtest_results,
            'is_profitable': backtest_results.get('is_profitable', False)
        }
        
    except Exception as e:
        logger.error(f"✗ Backtesting failed: {str(e)}")
        raise self.retry(exc=e, countdown=120)


@celery_app.task(bind=True, name='app.tasks.evaluate_and_save')
def evaluate_and_save(self, backtest_data):
    """
    Task 6: Evaluate results and save/delete strategy
    """
    logger.info(f"[Task 6/7] Evaluating results: {backtest_data['video_id']}")
    
    try:
        is_profitable = backtest_data['is_profitable']
        results = backtest_data['backtest_results']
        
        session = Session()
        
        if is_profitable:
            # Save profitable strategy
            logger.info("✓ Strategy is profitable - SAVING")
            session.execute(
                text("""
                    INSERT INTO proven_strategies 
                    (video_id, strategy_name, strategy_data, backtest_results, created_at)
                    VALUES (:vid, :name, :strategy, :results, :created)
                    ON CONFLICT (video_id) DO UPDATE 
                    SET backtest_results = :results, updated_at = :created
                """),
                {
                    'vid': backtest_data['video_id'],
                    'name': backtest_data['strategy'].get('strategy_name', 'Strategy'),
                    'strategy': str(backtest_data['strategy']),
                    'results': str(results),
                    'created': datetime.now()
                }
            )
        else:
            # Delete unprofitable strategy
            logger.info("✗ Strategy is not profitable - DELETING")
            session.execute(
                text("DELETE FROM ml_strategies WHERE video_id = :vid"),
                {'vid': backtest_data['video_id']}
            )
        
        # Update processed_videos status
        session.execute(
            text("""
                UPDATE processed_videos 
                SET status = :status, processed_at = :processed 
                WHERE video_id = :vid
            """),
            {
                'status': 'completed' if is_profitable else 'unprofitable',
                'processed': datetime.now(),
                'vid': backtest_data['video_id']
            }
        )
        
        session.commit()
        session.close()
        
        return {
            'status': 'success',
            'video_id': backtest_data['video_id'],
            'is_profitable': is_profitable,
            'win_rate': results.get('win_rate', 0),
            'profit_factor': results.get('profit_factor', 0),
            'action': 'saved' if is_profitable else 'deleted'
        }
        
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name='app.tasks.notify_completion')
def notify_completion(evaluation_result):
    """
    Task 7: Notify completion and log results
    """
    logger.info(f"[Task 7/7] Pipeline complete: {evaluation_result['video_id']}")
    
    if evaluation_result['is_profitable']:
        message = f"""
✅ PROFITABLE STRATEGY FOUND!
Video: {evaluation_result['video_id']}
Win Rate: {evaluation_result['win_rate']}%
Profit Factor: {evaluation_result['profit_factor']}
Action: {evaluation_result['action']}
        """
    else:
        message = f"""
❌ Strategy Not Profitable
Video: {evaluation_result['video_id']}
Win Rate: {evaluation_result['win_rate']}%
Action: {evaluation_result['action']}
        """
    
    logger.info(message)
    
    return {
        'status': 'completed',
        'message': message,
        'evaluation_result': evaluation_result
    }


# ==================== Workflow Chains ====================

@celery_app.task(name='app.tasks.process_video_pipeline')
def process_video_pipeline(video_id, file_path, filename):
    """
    Main pipeline: Chain all tasks together
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING PIPELINE: {video_id}")
    logger.info(f"{'='*60}\n")
    
    # Create task chain
    pipeline = chain(
        validate_video.s(video_id, file_path, filename),
        process_video.s(),
        extract_concepts.s(),
        generate_strategy.s(),
        backtest_strategy.s(),
        evaluate_and_save.s(),
        notify_completion.s()
    )
    
    # Execute chain
    result = pipeline.apply_async()
    
    return {
        'task_id': result.id,
        'video_id': video_id,
        'status': 'pipeline_started'
    }


# ==================== Scheduled Tasks ====================

@celery_app.task(name='app.tasks.scheduled_batch_processing')
def scheduled_batch_processing(batch_size=10):
    """
    Scheduled task: Process pending videos in batch
    """
    logger.info(f"Starting scheduled batch processing (batch_size={batch_size})")
    
    try:
        session = Session()
        
        # Get pending videos
        pending = session.execute(
            text("""
                SELECT video_id, filename 
                FROM processed_videos 
                WHERE status = 'uploaded' OR status IS NULL
                ORDER BY created_at ASC 
                LIMIT :limit
            """),
            {'limit': batch_size}
        ).fetchall()
        
        session.close()
        
        if not pending:
            logger.info("No pending videos to process")
            return {'status': 'no_videos'}
        
        logger.info(f"Found {len(pending)} pending videos")
        
        # Process each video
        for video_id, filename in pending:
            file_path = os.path.join(Config.VIDEO_WATCH_DIR, filename)
            process_video_pipeline.delay(video_id, file_path, filename)
        
        return {
            'status': 'success',
            'processed': len(pending)
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery_app.task(name='app.tasks.cleanup_old_results')
def cleanup_old_results(days=30):
    """
    Cleanup old results from database
    """
    logger.info(f"Cleaning up results older than {days} days")
    
    try:
        session = Session()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Delete old unprofitable strategies
        result = session.execute(
            text("""
                DELETE FROM processed_videos 
                WHERE status = 'unprofitable' 
                AND processed_at < :cutoff
            """),
            {'cutoff': cutoff_date}
        )
        
        deleted_count = result.rowcount
        session.commit()
        session.close()
        
        logger.info(f"Cleaned up {deleted_count} old results")
        
        return {
            'status': 'success',
            'deleted': deleted_count
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}