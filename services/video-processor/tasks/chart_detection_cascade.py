"""Cascade Chart Detection - Two-stage YOLO detection"""

import torch
from ultralytics import YOLO
import logging
import gc

logger = logging.getLogger(__name__)


class CascadeChartDetector:
    """
    Two-stage cascade detection:
    Stage 1: YOLOv8n (nano) - fast, low memory
    Stage 2: YOLOv8s (small) - accurate, high memory (only when uncertain)
    """

    def __init__(self, confidence_threshold=0.65, use_cascade=True):
        self.confidence_threshold = confidence_threshold
        self.use_cascade = use_cascade

        logger.info("Loading YOLOv8n (nano) model...")
        self.nano_model = YOLO("yolov8n.pt")

        self.base_model = None
        self.base_model_loaded = False

        self.stats = {
            'total_frames': 0,
            'nano_only': 0,
            'cascade_used': 0,
            'total_detections': 0,
            'nano_detections': 0,
            'base_detections': 0
        }

    def _load_base_model(self):
        if not self.base_model_loaded:
            logger.info("Loading YOLOv8s (small) model...")
            self.base_model = YOLO("yolov8s.pt")
            self.base_model_loaded = True

    def detect_charts(self, frames, batch_size=4):
        all_detections = []
        uncertain_frames = []

        logger.info(f"Starting cascade detection on {len(frames)} frames")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")

        logger.info("=" * 60)
        logger.info("STAGE 1: YOLOv8n (Nano) - Quick Scan")
        logger.info("=" * 60)

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.nano_model(
                batch_frames, half=True, device=0, verbose=False)

            for frame_idx, result in enumerate(results):
                actual_idx = i + frame_idx

                if len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = float(box.conf)
                        detection = {
                            'frame_index': actual_idx,
                            'confidence': confidence,
                            'bbox': box.xyxy.tolist(),
                            'model': 'nano',
                            'class': int(box.cls)
                        }

                        if confidence > self.confidence_threshold:
                            all_detections.append(detection)
                            self.stats['nano_detections'] += 1
                        else:
                            uncertain_frames.append(
                                (actual_idx, batch_frames[frame_idx]))

                    self.stats['nano_only'] += len(
                        [d for d in all_detections if d['model'] == 'nano'])

            if (i + batch_size) % (20 * batch_size) == 0:
                gc.collect()

        logger.info("Stage 1 Results:")
        logger.info(f"  High confidence detections: {len(all_detections)}")
        logger.info(f"  Frames marked for refinement: {len(uncertain_frames)}")
        self.stats['total_frames'] = len(frames)

        if self.use_cascade and len(uncertain_frames) > 0:
            logger.info("=" * 60)
            logger.info("STAGE 2: YOLOv8s (Small) - Detailed Refinement")
            logger.info("=" * 60)

            self._load_base_model()

            for frame_idx, frame in uncertain_frames:
                result = self.base_model(
                    frame, half=True, device=0, verbose=False)

                if len(result[0].boxes) > 0:
                    for box in result[0].boxes:
                        detection = {
                            'frame_index': frame_idx,
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist(),
                            'model': 'base',
                            'class': int(box.cls)
                        }
                        all_detections.append(detection)
                        self.stats['base_detections'] += 1
                        self.stats['cascade_used'] += 1

            logger.info("Stage 2 Results:")
            logger.info(
                f"  Base model detections: {self.stats['base_detections']}")
            logger.info(
                f"  Total cascade refinements: {self.stats['cascade_used']}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        self.stats['total_detections'] = len(all_detections)
        self._log_statistics()

        return all_detections, self.stats

    def _log_statistics(self):
        logger.info("=" * 60)
        logger.info("CASCADE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {self.stats['total_frames']}")
        logger.info(f"Nano-only detections: {self.stats['nano_only']}")
        logger.info(f"Base model detections: {self.stats['base_detections']}")
        logger.info(f"Total detections: {self.stats['total_detections']}")

        if self.stats['total_frames'] > 0:
            cascade_percent = (
                self.stats['cascade_used'] / self.stats['total_frames']) * 100
            logger.info(f"Frames using cascade: {cascade_percent:.1f}%")

        logger.info("=" * 60)
