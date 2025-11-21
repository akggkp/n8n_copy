"""
Video Processing Worker - Trading Education AI
Handles video upload, frame extraction, chart detection, and transcription
"""

import pika
import json
import cv2
import whisper
import torch
import numpy as np
import gc
import os
from pathlib import Path
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import psutil
import functools
from functools import partial

# Import cascade detector
from tasks.chart_detection_cascade import CascadeChartDetector

# Import keyword detector and clip generator
from keyword_detector import KeywordDetector, detect_keywords_from_whisper_result
from clip_generator import ClipGenerator, generate_clips_from_keyword_hits

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@postgres:5432/{os.getenv('POSTGRES_DB', 'trading_education')}"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,           # Max connections
    max_overflow=20,        # Extra temporary connections
    pool_recycle=3600,      # Recycle connections every hour
    pool_pre_ping=True      # Test connections before using
)
Base = declarative_base()
Session = sessionmaker(bind=engine)

from contextlib import contextmanager

@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

# Configuration
DELETE_VIDEO_AFTER_PROCESSING = os.getenv("DELETE_VIDEO_AFTER_PROCESSING", "true").lower() == "true"
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "300"))
USE_CASCADE = os.getenv("USE_CASCADE", "true").lower() == "true"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_CASCADE = os.getenv("WHISPER_CASCADE", "true").lower() == "true"
WHISPER_CASCADE_MODEL = os.getenv("WHISPER_CASCADE_MODEL", "large")

class ProcessedVideo(Base):
    __tablename__ = 'processed_videos'
    id = Column(Integer, primary_key=True)
    video_id = Column(String, unique=True, nullable=False)
    filename = Column(String, nullable=False)
    transcription = Column(Text)
    detected_charts = Column(JSON)
    key_concepts = Column(JSON)
    processing_stats = Column(JSON)
    processing_time_seconds = Column(Integer)
    processed_at = Column(DateTime, default=datetime.now)

# Base.metadata.create_all(engine)  # This should be handled by a separate migration script (e.g., using Alembic)

class MemoryManager:
    """Manage memory cleanup"""
    def __init__(self, cleanup_interval=300):
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = datetime.now()
    
    def should_cleanup(self):
        elapsed = (datetime.now() - self.last_cleanup).total_seconds()
        return elapsed > self.cleanup_interval
    
    def cleanup(self):
        logger.info("Starting memory cleanup...")
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        freed = mem_before - mem_after
        logger.info(f"Memory cleanup: {mem_before:.1f}MB → {mem_after:.1f}MB (freed {freed:.1f}MB)")
        self.last_cleanup = datetime.now()

class VideoProcessorCascade:
    def __init__(self):
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        self.whisper_cascade_model = None
        
        logger.info("Initializing cascade detector...")
        self.cascade_detector = CascadeChartDetector(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            use_cascade=USE_CASCADE
        )
        
        self.video_path = Path("/data/videos")
        self.output_path = Path("/data/processed")
        self.memory_manager = MemoryManager(cleanup_interval=MEMORY_CLEANUP_INTERVAL)
    
    def extract_frames(self, video_file, fps=1):
        """Extract frames with memory optimization"""
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_file}")
        
        frames = []
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, frame_rate // fps)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def extract_audio_and_transcribe(self, video_file):
        """Transcribe with optional cascade, returning full result with segments"""
        try:
            logger.info(f"Transcribing with Whisper {WHISPER_MODEL}...")
            # Get full result including segments for timestamped transcription
            result = self.whisper_model.transcribe(
                str(video_file), 
                language="en",
                word_timestamps=False  # Set to True for word-level timestamps if needed
            )
            text = result.get("text", "")
            
            if WHISPER_CASCADE:
                confidence = self._estimate_transcription_quality(text)
                logger.info(f"Transcription confidence: {confidence:.1%}")
                
                if confidence < 0.5:
                    logger.info("Confidence low, cascading to larger Whisper model...")
                    if self.whisper_cascade_model is None:
                        self.whisper_cascade_model = whisper.load_model(WHISPER_CASCADE_MODEL)
                    
                    result = self.whisper_cascade_model.transcribe(
                        str(video_file),
                        language="en",
                        word_timestamps=False
                    )
                    text = result.get("text", "")
                    logger.info("Using cascade model result")
            
            logger.info(f"Transcription completed: {len(text)} characters, {len(result.get('segments', []))} segments")
            return result  # Return full result with segments
        
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return {"text": "", "segments": []}
    
    def _estimate_transcription_quality(self, text):
        """Estimate if transcription has technical terms"""
        technical_terms = [
            'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr',
            'bullish', 'bearish', 'support', 'resistance', 'breakout',
            'crossover', 'divergence', 'reversal', 'momentum', 'trend'
        ]
        
        text_lower = text.lower()
        found_terms = sum(1 for term in technical_terms if term in text_lower)
        confidence = found_terms / len(technical_terms)
        return confidence
    
    def detect_charts(self, frames):
        """Detect charts using cascade"""
        detections, stats = self.cascade_detector.detect_charts(frames)
        if self.memory_manager.should_cleanup():
            self.memory_manager.cleanup()
        return detections, stats
    
    def delete_video_file(self, video_file):
        """Delete video file after processing"""
        try:
            if os.path.exists(video_file):
                file_size = os.path.getsize(video_file) / (1024 * 1024)
                os.remove(video_file)
                logger.info(f"Deleted video file: {video_file} ({file_size:.1f}MB)")
        except Exception as e:
            logger.error(f"Error deleting video: {str(e)}")
    
    def process_video(self, video_id, file_path):
        """Main processing pipeline with cascade, keyword detection, and clip generation"""
        start_time = datetime.now()
        
        try:
            video_file = Path(file_path)
            logger.info(f"Processing video: {video_id}")
            
            # Get video duration for clip generation
            clip_gen = ClipGenerator()
            video_duration = clip_gen.get_video_duration(str(video_file))
            
            # Extract frames
            frames = self.extract_frames(video_file, fps=1)
            
            # Transcribe audio - now returns full result with segments
            transcription_result = self.extract_audio_and_transcribe(video_file)
            transcription_text = transcription_result.get("text", "")
            transcription_segments = transcription_result.get("segments", [])
            
            # Detect keywords in transcription
            logger.info("Detecting trading keywords...")
            keyword_hits, keyword_stats = detect_keywords_from_whisper_result(
                transcription_result,
                video_duration
            )
            logger.info(f"Found {len(keyword_hits)} keyword hits: {keyword_stats.get('unique_keywords', 0)} unique keywords")
            
            # Generate clips for keyword hits
            logger.info("Generating clips for keyword hits...")
            clips_metadata = generate_clips_from_keyword_hits(
                str(video_file),
                video_id,
                keyword_hits,
                output_dir="/data/processed/clips",
                video_duration=video_duration
            )
            logger.info(f"Generated {len(clips_metadata)} clips")
            
            # Detect charts using CASCADE
            detections, cascade_stats = self.detect_charts(frames)
            
            # Cleanup after processing
            self.memory_manager.cleanup()
            
            # ✅ FIXED: Use 'with' statement for safe database session
            with session_scope() as session:
                processing_time = int((datetime.now() - start_time).total_seconds())

                # Store extended processing results
                extended_stats = cascade_stats.copy()
                extended_stats['keyword_hits_count'] = len(keyword_hits)
                extended_stats['clips_generated'] = len(clips_metadata)
                extended_stats['unique_keywords'] = keyword_stats.get('unique_keywords', 0)
                extended_stats['top_keywords'] = keyword_stats.get('top_keywords', [])[:5]
                extended_stats['video_duration'] = video_duration
                extended_stats['transcript_segments_count'] = len(transcription_segments)
                
                processed_video = ProcessedVideo(
                    video_id=video_id,
                    filename=video_file.name,
                    transcription=transcription_text,
                    detected_charts=detections,
                    key_concepts=[],
                    processing_stats=extended_stats,
                    processing_time_seconds=processing_time
                )
                session.add(processed_video)            # Session automatically closes here
            
            # Delete video file
            if DELETE_VIDEO_AFTER_PROCESSING:
                self.delete_video_file(file_path)
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.memory_manager.cleanup()
            
            logger.info(f"Video processing completed: {video_id}")
            
            # Return structured results for orchestrator
            return {
                'success': True,
                'video_id': video_id,
                'transcription': transcription_text,
                'transcription_segments': transcription_segments,
                'keyword_hits': keyword_hits,
                'keyword_stats': keyword_stats,
                'clips_metadata': clips_metadata,
                'detected_charts': detections,
                'processing_stats': extended_stats,
                'video_duration': video_duration
            }
        
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            return {'success': False, 'video_id': video_id, 'error': str(e)}
        
        finally:
            frames = []
            gc.collect()



def callback(ch, method, properties, body, processor):
    """RabbitMQ callback - reuses global processor"""
    try:
        message = json.loads(body.decode())
        video_id = message['video_id']
        file_path = message['file_path']
        logger.info(f"Received task: {video_id}")
        
        result = processor.process_video(video_id, file_path)

        if result.get('success', False) if isinstance(result, dict) else result:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Task completed successfully: {video_id}")
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            logger.error(f"Task failed: {video_id}")

        gc.collect()
        
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def main():
    RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")

    try:
        # Create a single instance of VideoProcessorCascade at startup
        logger.info("Initializing video processor...")
        processor = VideoProcessorCascade()

        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = connection.channel()
        channel.queue_declare(queue='video_processing', durable=True)
        channel.basic_qos(prefetch_count=1)

        # Use functools.partial to pass the processor instance to the callback
        partial_callback = functools.partial(callback, processor=processor)

        channel.basic_consume(
            queue='video_processing',
            on_message_callback=partial_callback
        )        
        logger.info("Video processor worker started with CASCADE approach")
        logger.info(f"Cascade enabled: {USE_CASCADE}")
        logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        
        channel.start_consuming()
        
    except Exception as e:
        logger.error(f"Worker error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
