import os
from datetime import timedelta

class Config:
    """Configuration for orchestrator service"""
    
    # Service Name
    SERVICE_NAME = "Trading AI Orchestrator"
    
    # Celery Configuration
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@rabbitmq:5672/')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://:password@redis:6379/0')
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'Asia/Kolkata'
    CELERY_ENABLE_UTC = True
    CELERY_TASK_TRACK_STARTED = True
    CELERY_TASK_TIME_LIMIT = 3600  # 1 hour max per task
    CELERY_TASK_SOFT_TIME_LIMIT = 3000  # 50 minutes soft limit
    
    # Task retry configuration
    CELERY_TASK_AUTORETRY_FOR = (Exception,)
    CELERY_TASK_RETRY_KWARGS = {'max_retries': 3}
    CELERY_TASK_DEFAULT_RETRY_DELAY = 60  # 1 minute
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://tradingai:password@postgres:5432/trading_education')
    # Service URLs
    VIDEO_PROCESSOR_URL = os.getenv('VIDEO_PROCESSOR_URL', 'http://video-processor:8000')
    ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://ml-service:8002')
    BACKTESTING_SERVICE_URL = os.getenv('BACKTESTING_SERVICE_URL', 'http://backtesting-service:8001')
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
    
    # Video Processing
    VIDEO_WATCH_DIR = os.getenv('VIDEO_WATCH_DIR', '/data/videos')
    PROCESSED_DIR = os.getenv('PROCESSED_DIR', '/data/processed')
    
    # Cascade Configuration
    USE_CASCADE = os.getenv('USE_CASCADE', 'true').lower() == 'true'
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.65'))
    
    # Backtesting Thresholds
    MIN_WIN_RATE = float(os.getenv('MIN_WIN_RATE_TO_SAVE', '55'))
    MIN_PROFIT_FACTOR = float(os.getenv('MIN_PROFIT_FACTOR', '1.5'))
    
    # Flask Web UI
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '8080'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Scheduled Batch Processing
    BATCH_SCHEDULE_HOUR = int(os.getenv('BATCH_SCHEDULE_HOUR', '2'))  # 2 AM
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
    # File Watcher
    ENABLE_FILE_WATCHER = os.getenv('ENABLE_FILE_WATCHER', 'true').lower() == 'true'
    FILE_WATCHER_PATTERNS = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    # LLM Configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))