"""File watcher to automatically process new videos"""
from app.tasks import process_video_pipeline
from config import Config
import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class VideoFileHandler(FileSystemEventHandler):
    """Handler for new video files"""

    def __init__(self):
        super().__init__()
        self.processing = set()  # Track files being processed
        self.cooldown = 5  # Wait 5 seconds after file is created

    def is_video_file(self, filename):
        """Check if file is a video"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    def on_created(self, event):
        """Handle new file creation"""
        if event.is_directory:
            return

        file_path = event.src_path
        filename = os.path.basename(file_path)

        # Check if it's a video file
        if not self.is_video_file(filename):
            return

        # Avoid duplicate processing
        if file_path in self.processing:
            return

        logger.info(f"📹 New video detected: {filename}")

        # Add to processing set
        self.processing.add(file_path)

        # Wait for file to finish writing
        time.sleep(self.cooldown)

        try:
            # Generate video ID
            video_id = f"video_{uuid.uuid4().hex[:12]}"

            # Trigger processing pipeline
            logger.info(f"🚀 Starting pipeline for: {video_id}")
            process_video_pipeline.delay(video_id, file_path, filename)

            logger.info(f"✓ Pipeline triggered for: {filename}")

        except Exception as e:
            logger.error(f"✗ Failed to trigger pipeline: {str(e)}")
            self.processing.remove(file_path)

        # Remove from processing after 60 seconds
        time.sleep(60)
        if file_path in self.processing:
            self.processing.remove(file_path)


def start_file_watcher():
    """Start watching for new videos"""

    if not Config.ENABLE_FILE_WATCHER:
        logger.info("File watcher disabled in config")
        return

    watch_dir = Config.VIDEO_WATCH_DIR

    # Create directory if it doesn't exist
    os.makedirs(watch_dir, exist_ok=True)

    logger.info(f"📂 Starting file watcher on: {watch_dir}")
    logger.info(f"   Looking for: {', '.join(Config.FILE_WATCHER_PATTERNS)}")

    event_handler = VideoFileHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()

    logger.info("✓ File watcher started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()

    observer.join()
    logger.info("File watcher stopped")


if __name__ == '__main__':
    start_file_watcher()
