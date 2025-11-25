# orchestrator/app/tasks.py - COMPLETE VERSION
from celery import shared_task, chain
from celery.utils.log import get_task_logger
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from app.database import SessionLocal
from app.models import MediaItem, Transcript, KeywordHit, Clip, ProvenStrategy

logger = get_task_logger(__name__)

# Service URLs
VIDEO_PROCESSOR_URL = os.getenv(
    'VIDEO_PROCESSOR_URL',
    'http://video-processor:8000')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://ml-service:8002')
API_SERVICE_URL = os.getenv('API_SERVICE_URL', 'http://api-service:8003')
EMBEDDINGS_SERVICE_URL = os.getenv(
    'EMBEDDINGS_SERVICE_URL',
    'http://embeddings-service:8004')
BACKTEST_SERVICE_URL = os.getenv(
    'BACKTEST_SERVICE_URL',
    'http://backtesting-service:8001')


def get_retry_session():
    """Get HTTP session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


@shared_task(bind=True, name='app.tasks.validate_video')
def validate_video(self, media_item_id, file_path, filename):
    """Validate video file exists and is accessible"""
    try:
        logger.info(
            f"Validating video {filename} for media_item {media_item_id}")

        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("Video file is empty")

        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        if not any(file_path.lower().endswith(ext)
                   for ext in valid_extensions):
            raise ValueError(
                f"Invalid file format. Supported: {valid_extensions}")

        # Update database
        db = SessionLocal()
        try:
            media_item = db.query(MediaItem).filter_by(
                id=media_item_id).first()
            if media_item:
                media_item.status = 'validated'
                media_item.file_size = file_size
                db.commit()
        finally:
            db.close()

        logger.info(f"Video validated successfully: {filename}")
        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'file_path': file_path,
            'file_size': file_size
        }

    except Exception as e:
        logger.error(f"Video validation failed: {str(e)}")

        # Update status to failed
        db = SessionLocal()
        try:
            media_item = db.query(MediaItem).filter_by(
                id=media_item_id).first()
            if media_item:
                media_item.status = 'failed'
                media_item.error_message = str(e)
                db.commit()
        finally:
            db.close()

        raise


def _update_media_item_status(db, media_item_id, status, error_message=None):
    """Helper to update media item status in the database."""
    try:
        media_item = db.query(MediaItem).filter_by(id=media_item_id).first()
        if media_item:
            media_item.status = status
            if error_message:
                media_item.error_message = error_message
            db.commit()
    except Exception as e:
        logger.error(f"Error updating status for media item {media_item_id}: {e}")
        db.rollback()


def _call_video_processor(file_path):
    """Calls the video processor service and returns the result."""
    session = get_retry_session()
    response = session.post(
        f"{VIDEO_PROCESSOR_URL}/process",
        json={"file_path": file_path},
        timeout=600  # 10 minutes for large videos
    )
    response.raise_for_status()
    result = response.json()
    if result.get('status') != 'success':
        raise ValueError("Video processing failed in the external service.")
    return result


def _save_processing_results(db, media_item_id, result):
    """Saves transcript and metadata from video processing."""
    transcript_segments = result.get('transcript', [])
    if not transcript_segments:
        logger.warning("No transcript segments returned")

    for segment in transcript_segments:
        transcript = Transcript(
            media_item_id=media_item_id,
            start_time=segment['start_time'],
            end_time=segment['end_time'],
            text=segment['text']
        )
        db.add(transcript)

    media_item = db.query(MediaItem).filter_by(id=media_item_id).first()
    if media_item:
        media_item.duration = result.get('metadata', {}).get('duration')
        media_item.frame_count = result.get('metadata', {}).get('total_frames')

    db.commit()
    logger.info(f"Saved {len(transcript_segments)} transcript segments")
    return len(transcript_segments)


@shared_task(bind=True, name='app.tasks.process_video')
def process_video(self, previous_result, file_path):
    """Process video: extract audio, transcribe, extract frames"""
    media_item_id = previous_result.get('media_item_id')
    logger.info(f"Processing video for media_item {media_item_id}")
    db = SessionLocal()
    try:
        _update_media_item_status(db, media_item_id, 'processing')

        result = _call_video_processor(file_path)

        transcript_count = _save_processing_results(db, media_item_id, result)

        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'transcript_count': transcript_count,
            'frames': result.get('frames', []),
            'metadata': result.get('metadata', {})
        }

    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        _update_media_item_status(db, media_item_id, 'failed', error_message=str(e))
        raise
    finally:
        db.close()


@shared_task(bind=True, name='app.tasks.detect_keywords')
def detect_keywords(self, previous_result):
    """Detect trading keywords in transcripts"""
    try:
        media_item_id = previous_result.get('media_item_id')
        logger.info(f"Detecting keywords for media_item {media_item_id}")

        # Get transcripts from database
        db = SessionLocal()
        try:
            transcripts = db.query(Transcript).filter_by(
                media_item_id=media_item_id).all()

            if not transcripts:
                logger.warning("No transcripts found")
                return {
                    'status': 'success',
                    'media_item_id': media_item_id,
                    'keywords_found': 0
                }

            # Combine all transcript text
            full_transcript = " ".join([t.text for t in transcripts])

            # Call ML service for concept extraction
            session = get_retry_session()
            response = session.post(
                f"{ML_SERVICE_URL}/extract_concepts",
                json={"transcript": full_transcript},
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            keywords = result.get('keywords', [])

            # Save keywords to database
            for keyword_data in keywords:
                # Find which transcript segment contains this keyword
                for transcript in transcripts:
                    if keyword_data['keyword'].lower(
                    ) in transcript.text.lower():
                        keyword_hit = KeywordHit(
                            media_item_id=media_item_id,
                            keyword=keyword_data['keyword'],
                            category=keyword_data['category'],
                            start_time=transcript.start_time,
                            end_time=transcript.end_time,
                            confidence=keyword_data['confidence'],
                            context_text=keyword_data['context']
                        )
                        db.add(keyword_hit)
                        break  # Found in this transcript

            db.commit()
            logger.info(f"Saved {len(keywords)} keyword hits")

            return {
                'status': 'success',
                'media_item_id': media_item_id,
                'keywords_found': len(keywords)
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Keyword detection failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.generate_clips')
def generate_clips(self, previous_result, file_path):
    """Generate video clips for keyword hits"""
    try:
        media_item_id = previous_result.get('media_item_id')
        logger.info(f"Generating clips for media_item {media_item_id}")

        import subprocess
        from pathlib import Path

        # Get keyword hits from database
        db = SessionLocal()
        try:
            keyword_hits = db.query(KeywordHit).filter_by(
                media_item_id=media_item_id).all()

            if not keyword_hits:
                logger.warning("No keyword hits found")
                return {
                    'status': 'success',
                    'media_item_id': media_item_id,
                    'clips_created': 0
                }

            clips_dir = Path(
                os.getenv(
                    'CLIPS_OUTPUT_DIR',
                    '/data/processed/clips'))
            clips_dir.mkdir(parents=True, exist_ok=True)

            clips_created = 0

            for hit in keyword_hits[:20]:  # Limit to 20 clips per video
                try:
                    # Generate clip filename
                    clip_filename = f"{media_item_id}_{hit.id}_{hit.keyword.replace(' ', '_')}.mp4"
                    clip_path = clips_dir / clip_filename

                    # Calculate clip duration (add 2 seconds buffer)
                    start_time = max(0, hit.start_time - 1)
                    end_time = hit.end_time + 1
                    duration = end_time - start_time

                    # Use FFmpeg to extract clip
                    cmd = [
                        'ffmpeg',
                        '-i', file_path,
                        '-ss', str(start_time),
                        '-t', str(duration),
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-strict', 'experimental',
                        str(clip_path),
                        '-y'
                    ]

                    subprocess.run(cmd, check=True, capture_output=True)

                    # Save clip to database
                    clip = Clip(
                        media_item_id=media_item_id,
                        keyword_hit_id=hit.id,
                        file_path=str(clip_path),
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration
                    )
                    db.add(clip)
                    clips_created += 1

                except Exception as e:
                    logger.error(
                        f"Failed to create clip for keyword {hit.keyword}: {str(e)}")
                    continue

            db.commit()
            logger.info(f"Created {clips_created} clips")

            return {
                'status': 'success',
                'media_item_id': media_item_id,
                'clips_created': clips_created
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Clip generation failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.extract_concepts')
def extract_concepts(self, previous_result):
    """Extract trading concepts (already done in detect_keywords)"""
    # Concepts are extracted together with keywords
    # This task exists for pipeline consistency
    logger.info("Concepts extracted (combined with keyword detection)")
    return {
        'status': 'success',
        'media_item_id': previous_result.get('media_item_id'),
        'note': 'Concepts extracted in keyword detection phase'
    }


@shared_task(bind=True, name='app.tasks.generate_embeddings')
def generate_embeddings(self, previous_result):
    """Generate embeddings for transcripts and clips"""
    try:
        media_item_id = previous_result.get('media_item_id')
        logger.info(f"Generating embeddings for media_item {media_item_id}")

        # Get transcripts
        db = SessionLocal()
        try:
            transcripts = db.query(Transcript).filter_by(
                media_item_id=media_item_id).all()

            if not transcripts:
                logger.warning("No transcripts found")
                return {
                    'status': 'success',
                    'media_item_id': media_item_id,
                    'embeddings_created': 0
                }

            # Prepare segments for embedding service
            segments = [
                {
                    'id': t.id,
                    'text': t.text,
                    'start_time': t.start_time,
                    'end_time': t.end_time
                }
                for t in transcripts
            ]

            # Call embeddings service
            session = get_retry_session()
            response = session.post(
                f"{EMBEDDINGS_SERVICE_URL}/embed",
                json={
                    'media_item_id': media_item_id,
                    'embedding_type': 'transcript',
                    'segments': segments
                },
                timeout=300
            )
            response.raise_for_status()

            result = response.json()

            logger.info(
                f"Generated {result.get('embeddings_created', 0)} embeddings")

            return {
                'status': 'success',
                'media_item_id': media_item_id,
                'embeddings_created': result.get('embeddings_created', 0)
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.extract_features')
def extract_features(self, previous_result):
    """Extract ML features from keywords and embeddings"""
    try:
        media_item_id = previous_result.get('media_item_id')
        logger.info(f"Extracting features for media_item {media_item_id}")

        db = SessionLocal()
        try:
            # Get keyword hits
            keyword_hits = db.query(KeywordHit).filter_by(
                media_item_id=media_item_id).all()

            if not keyword_hits:
                logger.warning("No keyword hits found")
                return {
                    'status': 'success',
                    'media_item_id': media_item_id,
                    'features': {}
                }

            # Calculate features
            from collections import Counter

            # Category distribution
            category_counts = Counter([kw.category for kw in keyword_hits])

            # Most frequent keywords
            keyword_counts = Counter([kw.keyword for kw in keyword_hits])
            top_keywords = keyword_counts.most_common(10)

            # Timing analysis
            timestamps = [kw.start_time for kw in keyword_hits]
            avg_timestamp = sum(timestamps) / \
                len(timestamps) if timestamps else 0

            features = {
                'total_keywords': len(keyword_hits),
                'unique_keywords': len(keyword_counts),
                'category_distribution': dict(category_counts),
                'top_keywords': top_keywords,
                'avg_keyword_timestamp': avg_timestamp,
                'keyword_density': len(keyword_hits) / (previous_result.get('duration', 1) or 1)
            }

            logger.info(f"Extracted {len(features)} feature groups")

            return {
                'status': 'success',
                'media_item_id': media_item_id,
                'features': features
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.generate_strategy')
def generate_strategy(self, previous_result):
    """Generate trading strategy from features"""
    try:
        media_item_id = previous_result.get('media_item_id')
        features = previous_result.get('features', {})

        logger.info(f"Generating strategy for media_item {media_item_id}")

        # Basic strategy generation based on features
        category_dist = features.get('category_distribution', {})

        # Determine primary focus
        if category_dist.get(
                'technical_indicator',
                0) > category_dist.get(
                'price_action',
                0):
            strategy_type = 'indicator_based'
        else:
            strategy_type = 'price_action'

        # Generate simple strategy rules
        strategy_rules = {
            'type': strategy_type,
            'timeframe': '1h',  # Default timeframe
            'entry_conditions': [],
            'exit_conditions': [],
            'risk_reward_ratio': 2.0
        }

        # Add entry conditions based on top keywords
        top_keywords = features.get('top_keywords', [])
        for keyword, count in top_keywords[:3]:
            if 'rsi' in keyword.lower():
                strategy_rules['entry_conditions'].append(
                    'RSI < 30 (oversold)')
            elif 'support' in keyword.lower():
                strategy_rules['entry_conditions'].append(
                    'Price near support level')
            elif 'breakout' in keyword.lower():
                strategy_rules['entry_conditions'].append(
                    'Breakout confirmation')

        # Add exit conditions
        if category_dist.get('risk_management', 0) > 0:
            strategy_rules['exit_conditions'].append('Stop loss: 2%')
            strategy_rules['exit_conditions'].append('Take profit: 4%')

        logger.info(
            f"Generated {strategy_type} strategy with {len(strategy_rules['entry_conditions'])} entry conditions")

        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'strategy': strategy_rules
        }

    except Exception as e:
        logger.error(f"Strategy generation failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.backtest_strategy')
def backtest_strategy(self, previous_result):
    """Backtest the generated strategy"""
    try:
        media_item_id = previous_result.get('media_item_id')
        strategy = previous_result.get('strategy', {})

        logger.info(f"Backtesting strategy for media_item {media_item_id}")

        # Call backtesting service
        session = get_retry_session()
        response = session.post(
            f"{BACKTEST_SERVICE_URL}/backtest",
            json={
                'media_item_id': media_item_id,
                'strategy': strategy,
                'start_date': '2023-01-01',
                'end_date': '2024-01-01',
                'initial_capital': 10000
            },
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        metrics = result.get('metrics', {})

        logger.info(
            f"Backtest complete - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'backtest_metrics': metrics,
            'strategy': strategy
        }

    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.evaluate_and_promote')
def evaluate_and_promote(self, previous_result):
    """Evaluate backtest results and promote if passing thresholds"""
    try:
        media_item_id = previous_result.get('media_item_id')
        metrics = previous_result.get('backtest_metrics', {})
        previous_result.get('strategy', {})

        logger.info(f"Evaluating strategy for media_item {media_item_id}")

        # Get thresholds from environment
        min_sharpe = float(os.getenv('MIN_SHARPE_RATIO', '1.0'))
        min_win_rate = float(os.getenv('MIN_WIN_RATE_PERCENT', '55'))
        max_drawdown = float(os.getenv('MAX_DRAWDOWN_PERCENT', '25'))

        # Evaluate metrics
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0) * 100
        drawdown = abs(metrics.get('max_drawdown', 100))

        passing = (
            sharpe_ratio >= min_sharpe and
            win_rate >= min_win_rate and
            drawdown <= max_drawdown
        )

        if passing:
            # Save to proven_strategies table
            db = SessionLocal()
            try:
                proven_strategy = ProvenStrategy(
                    media_item_id=media_item_id,
                    sharpe_ratio=sharpe_ratio,
                    win_rate=win_rate / 100,  # Store as decimal
                    max_drawdown=drawdown / 100,
                    total_return=metrics.get('total_return', 0),
                    num_trades=metrics.get('num_trades', 0),
                    avg_trade_duration=metrics.get('avg_trade_duration', 0),
                    status='promoted'
                )
                db.add(proven_strategy)
                db.commit()

                logger.info(
                    f"Strategy promoted! Sharpe: {sharpe_ratio:.2f}, Win Rate: {win_rate:.1f}%")

            finally:
                db.close()
        else:
            logger.info(
                f"Strategy did not pass thresholds. Sharpe: {sharpe_ratio:.2f}, Win Rate: {win_rate:.1f}%")

        # Update media item status
        db = SessionLocal()
        try:
            media_item = db.query(MediaItem).filter_by(
                id=media_item_id).first()
            if media_item:
                media_item.status = 'completed'
                db.commit()
        finally:
            db.close()

        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'promoted': passing,
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


@shared_task(bind=True, name='app.tasks.run_full_pipeline')
def run_full_pipeline(self, media_item_id, file_path, filename):
    """Run complete video processing pipeline"""
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
            extract_features.s(),
            generate_strategy.s(),
            backtest_strategy.s(),
            evaluate_and_promote.s()
        )

        # Execute pipeline asynchronously
        result = pipeline.apply_async()

        return {
            'status': 'pipeline_started',
            'media_item_id': media_item_id,
            'task_id': result.id
        }

    except Exception as e:
        logger.error(f"Pipeline start failed: {str(e)}")
        raise
