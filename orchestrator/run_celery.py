import os
import sys
from celery.schedules import crontab

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.celery_app import celery_app
from config import Config

celery_app.conf.update(
    broker_url=Config.CELERY_BROKER_URL,
    result_backend=Config.CELERY_RESULT_BACKEND,
    task_serializer=Config.CELERY_TASK_SERIALIZER,
    result_serializer=Config.CELERY_RESULT_SERIALIZER,
    accept_content=Config.CELERY_ACCEPT_CONTENT,
    timezone=Config.CELERY_TIMEZONE,
    enable_utc=Config.CELERY_ENABLE_UTC,
    task_track_started=Config.CELERY_TASK_TRACK_STARTED,
    task_time_limit=Config.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=Config.CELERY_TASK_SOFT_TIME_LIMIT,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
)

celery_app.conf.beat_schedule = {
    'batch-process-videos': {
        'task': 'app.tasks.scheduled_batch_processing',
        'schedule': crontab(hour=Config.BATCH_SCHEDULE_HOUR, minute=0),
        'args': (Config.BATCH_SIZE,)
    },
    'cleanup-old-results': {
        'task': 'app.tasks.cleanup_old_results',
        'schedule': crontab(hour=3, minute=0),
        'args': (30,)
    }
}
