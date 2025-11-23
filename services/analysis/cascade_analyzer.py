"""
Cascade Detection Analyzer - Analyze cascade detection performance
"""

import json
import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CascadeAnalyzer:
    """Analyze cascade detection statistics from database"""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Connect to database"""
        try:
            from sqlalchemy import create_engine
            self.engine = create_engine(self.db_url)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    def get_cascade_stats(self, hours: int = 24) -> Dict:
        """Get cascade statistics for last N hours"""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker

            engine = create_engine(self.db_url)
            Session = sessionmaker(bind=engine)
            session = Session()

            query = text("""
                SELECT
                    pv.video_id,
                    pv.filename,
                    pv.processing_time_seconds,
                    pv.processing_stats
                FROM processed_videos pv
                WHERE pv.processed_at > NOW() - INTERVAL ':hours hours'
                ORDER BY pv.processed_at DESC;
            """)

            results = session.execute(query, {"hours": hours}).fetchall()

            stats = {
                'total_videos': len(results),
                'total_frames': 0,
                'total_nano_only': 0,
                'total_cascade_used': 0,
                'total_detections': 0,
                'avg_processing_time': 0,
                'cascade_usage_percent': 0,
                'videos': []
            }

            processing_times = []

            for video_id, filename, proc_time, proc_stats in results:
                stats_dict = json.loads(proc_stats) if proc_stats else {}

                stats['total_frames'] += stats_dict.get('total_frames', 0)
                stats['total_nano_only'] += stats_dict.get('nano_only', 0)
                stats['total_cascade_used'] += stats_dict.get(
                    'cascade_used', 0)
                stats['total_detections'] += stats_dict.get(
                    'total_detections', 0)
                processing_times.append(proc_time)

                stats['videos'].append({
                    'video_id': video_id,
                    'filename': filename,
                    'processing_time': proc_time,
                    'stats': stats_dict
                })

            if processing_times:
                stats['avg_processing_time'] = sum(
                    processing_times) / len(processing_times)

            if stats['total_frames'] > 0:
                stats['cascade_usage_percent'] = (
                    stats['total_cascade_used'] / stats['total_frames']
                ) * 100

            session.close()
            return stats

        except Exception as e:
            logger.error(f"Error fetching cascade stats: {str(e)}")
            return {}

    def get_detection_summary(self) -> Dict:
        """Get summary of all detections"""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker

            engine = create_engine(self.db_url)
            Session = sessionmaker(bind=engine)
            session = Session()

            query = text("""
                SELECT
                    COUNT(*) as total_processed,
                    AVG((processing_stats->>'total_frames')::INT) as avg_frames,
                    AVG((processing_stats->>'total_detections')::INT) as avg_detections,
                    AVG(processing_time_seconds) as avg_time
                FROM processed_videos
                WHERE processing_stats IS NOT NULL;
            """)

            result = session.execute(query).fetchone()
            session.close()

            if result:
                return {
                    'total_processed': result[0], 'avg_frames_per_video': float(
                        result[1]) if result[1] else 0, 'avg_detections_per_video': float(
                        result[2]) if result[2] else 0, 'avg_processing_time': float(
                        result[3]) if result[3] else 0}

            return {}

        except Exception as e:
            logger.error(f"Error fetching detection summary: {str(e)}")
            return {}

    def print_report(self, hours: int = 24):
        """Print cascade performance report"""
        stats = self.get_cascade_stats(hours)
        summary = self.get_detection_summary()

        print("\n" + "=" * 70)
        print(f"CASCADE DETECTION ANALYSIS (Last {hours} hours)")
        print("=" * 70)

        if stats:
            print(f"Total videos processed: {stats['total_videos']}")
            print(f"Total frames processed: {stats['total_frames']}")
            print(f"Nano-only detections: {stats['total_nano_only']}")
            print(f"Cascade refinements: {stats['total_cascade_used']}")
            print(f"Total detections: {stats['total_detections']}")
            print(
                f"Average processing time: {stats['avg_processing_time']:.1f}s")
            print(f"Cascade usage: {stats['cascade_usage_percent']:.1f}%")

        print("\nOverall Summary:")
        print("-" * 70)
        if summary:
            print(f"Total processed (all time): {summary['total_processed']}")
            print(
                f"Average frames per video: {summary['avg_frames_per_video']:.1f}")
            print(
                f"Average detections per video: {summary['avg_detections_per_video']:.1f}")
            print(
                f"Average processing time: {summary['avg_processing_time']:.1f}s")

        # Per-video breakdown
        if stats.get('videos'):
            print("\nPer-Video Breakdown (Last 5):")
            print("-" * 70)
            for video in stats['videos'][:5]:
                v_stats = video['stats']
                cascade_pct = (
                    v_stats.get('cascade_used', 0) / v_stats.get('total_frames', 1)
                ) * 100 if v_stats.get('total_frames', 0) > 0 else 0

                print(f"\nVideo: {video['filename']}")
                print(f"  Processing time: {video['processing_time']}s")
                print(f"  Frames: {v_stats.get('total_frames', 0)}")
                print(f"  Detections: {v_stats.get('total_detections', 0)}")
                print(f"  Cascade usage: {cascade_pct:.1f}%")

        print("\n" + "=" * 70 + "\n")


def main():
    """Entry point"""
    analyzer = CascadeAnalyzer()
    analyzer.print_report(hours=24)


if __name__ == "__main__":
    main()
