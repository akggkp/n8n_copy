# clip_generator.py
# Module for extracting video clips around keyword timestamps using ffmpeg

import subprocess
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class ClipMetadata:
    """Metadata for generated video clip"""
    clip_id: str
    video_id: str
    keyword: str
    start_time: float
    end_time: float
    duration_seconds: float
    file_path: str
    file_size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ClipGenerator:
    """Generates video clips around detected keywords using ffmpeg"""
    
    def __init__(
        self,
        output_dir: str,
        ffmpeg_path: str = "ffmpeg",
        context_padding: float = 5.0,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
        preset: str = "fast"
    ):
        """
        Initialize ClipGenerator
        
        Args:
            output_dir: Base directory for clip output
            ffmpeg_path: Path to ffmpeg executable
            context_padding: Seconds to add before/after keyword hit (default 5 sec)
            video_codec: FFmpeg video codec (default libx264)
            audio_codec: FFmpeg audio codec (default aac)
            crf: Quality (0-51, lower=better, default 23)
            preset: Encoding preset (ultrafast/superfast/veryfast/faster/fast/medium/slow/slower)
        """
        self.output_dir = Path(output_dir)
        self.ffmpeg_path = ffmpeg_path
        self.context_padding = context_padding
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.crf = crf
        self.preset = preset
        
        # Create output directory if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_clip(
        self,
        video_path: str,
        video_id: str,
        keyword: str,
        start_time: float,
        end_time: float,
        clip_index: int = 0,
        include_audio: bool = True
    ) -> Optional[ClipMetadata]:
        """
        Generate a video clip around keyword timestamp
        
        Args:
            video_path: Path to source video file
            video_id: Unique video identifier
            keyword: Keyword name for clip labeling
            start_time: Start time of keyword hit (seconds)
            end_time: End time of keyword hit (seconds)
            clip_index: Index for multiple clips from same keyword
            include_audio: Whether to include audio in clip
        
        Returns:
            ClipMetadata object or None if generation failed
        """
        try:
            # Calculate clip boundaries with padding
            clip_start = max(0, start_time - self.context_padding)
            clip_end = end_time + self.context_padding
            clip_duration = clip_end - clip_start
            
            # Create output directory for this video
            video_clip_dir = self.output_dir / video_id
            video_clip_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate safe filename
            safe_keyword = "".join(c for c in keyword if c.isalnum() or c in ('-', '_')).rstrip()
            clip_filename = f"{video_id}_{safe_keyword}_{clip_index}.mp4"
            clip_path = video_clip_dir / clip_filename
            
            # Build ffmpeg command
            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-ss", str(clip_start),
                "-to", str(clip_end),
                "-c:v", self.video_codec,
                "-crf", str(self.crf),
                "-preset", self.preset,
            ]
            
            # Add audio handling
            if include_audio:
                cmd.extend(["-c:a", self.audio_codec])
            else:
                cmd.append("-an")
            
            cmd.extend([
                "-y",  # Overwrite output file
                str(clip_path)
            ])
            
            logger.info(f"Generating clip: {keyword} from {clip_start:.2f}s to {clip_end:.2f}s")
            
            # Execute ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
            
            # Get file metadata
            file_size = clip_path.stat().st_size if clip_path.exists() else 0
            video_info = self._get_video_info(str(clip_path))
            
            metadata = ClipMetadata(
                clip_id=f"{video_id}_{safe_keyword}_{clip_index}",
                video_id=video_id,
                keyword=keyword,
                start_time=clip_start,
                end_time=clip_end,
                duration_seconds=clip_duration,
                file_path=str(clip_path),
                file_size_bytes=file_size,
                width=video_info.get('width'),
                height=video_info.get('height')
            )
            
            logger.info(f"Clip generated successfully: {clip_path}")
            return metadata
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout for clip: {keyword}")
            return None
        except Exception as e:
            logger.error(f"Error generating clip: {str(e)}")
            return None
    
    def generate_clips_batch(
        self,
        video_path: str,
        video_id: str,
        keyword_hits: List[Dict],
        include_audio: bool = True
    ) -> List[ClipMetadata]:
        """
        Generate multiple clips from keyword hits
        
        Args:
            video_path: Path to source video file
            video_id: Unique video identifier
            keyword_hits: List of dicts with 'keyword', 'start_time', 'end_time'
            include_audio: Whether to include audio
        
        Returns:
            List of ClipMetadata objects for successfully generated clips
        """
        clips = []
        
        for idx, hit in enumerate(keyword_hits):
            clip_meta = self.generate_clip(
                video_path=video_path,
                video_id=video_id,
                keyword=hit['keyword'],
                start_time=hit['start_time'],
                end_time=hit['end_time'],
                clip_index=idx,
                include_audio=include_audio
            )
            
            if clip_meta:
                clips.append(clip_meta)
        
        logger.info(f"Batch generation complete: {len(clips)}/{len(keyword_hits)} clips created")
        return clips
    
    def extract_keyframe(
        self,
        video_path: str,
        video_id: str,
        keyword: str,
        timestamp: float,
        frame_index: int = 0
    ) -> Optional[Dict]:
        """
        Extract a single keyframe at specific timestamp
        
        Args:
            video_path: Path to source video file
            video_id: Unique video identifier
            keyword: Keyword name for frame labeling
            timestamp: Time in seconds to extract frame
            frame_index: Index for multiple frames from same keyword
        
        Returns:
            Dict with frame metadata or None if extraction failed
        """
        try:
            # Create output directory for frames
            frame_dir = self.output_dir / video_id / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate safe filename
            safe_keyword = "".join(c for c in keyword if c.isalnum() or c in ('-', '_')).rstrip()
            frame_filename = f"{video_id}_{safe_keyword}_{frame_index}.jpg"
            frame_path = frame_dir / frame_filename
            
            # Build ffmpeg command for frame extraction
            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-ss", str(timestamp),
                "-vframes", "1",
                "-q:v", "2",  # Quality 1-31 (lower is better)
                "-y",
                str(frame_path)
            ]
            
            logger.info(f"Extracting keyframe: {keyword} at {timestamp:.2f}s")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg frame extraction error: {result.stderr}")
                return None
            
            # Get frame dimensions
            video_info = self._get_video_info(str(video_path))
            
            frame_metadata = {
                'frame_id': f"{video_id}_{safe_keyword}_{frame_index}",
                'video_id': video_id,
                'keyword': keyword,
                'timestamp': timestamp,
                'file_path': str(frame_path),
                'file_size_bytes': frame_path.stat().st_size if frame_path.exists() else 0,
                'width': video_info.get('width'),
                'height': video_info.get('height'),
                'contains_chart': False  # Will be set by chart detection service
            }
            
            logger.info(f"Keyframe extracted: {frame_path}")
            return frame_metadata
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout for frame extraction: {keyword}")
            return None
        except Exception as e:
            logger.error(f"Error extracting keyframe: {str(e)}")
            return None
    
    def _get_video_info(self, video_path: str) -> Dict:
        """
        Get video dimensions and basic info using ffprobe
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dict with 'width', 'height', 'duration' keys
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "default=noprint_wrappers=1:nokey=1:ch=,",
                str(video_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                return {
                    'width': int(lines[0]),
                    'height': int(lines[1]),
                    'duration': float(lines[2]) if len(lines) > 2 else None
                }
        except Exception as e:
            logger.warning(f"Could not get video info: {str(e)}")
        
        return {'width': None, 'height': None, 'duration': None}
    
    def cleanup_clips(self, video_id: str) -> bool:
        """
        Delete all clips for a specific video
        
        Args:
            video_id: Unique video identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            video_clip_dir = self.output_dir / video_id
            if video_clip_dir.exists():
                import shutil
                shutil.rmtree(video_clip_dir)
                logger.info(f"Cleaned up clips for video: {video_id}")
                return True
        except Exception as e:
            logger.error(f"Error cleaning up clips: {str(e)}")
            return False
        
        return False


def format_clips_for_database(clips: List[ClipMetadata]) -> List[Dict]:
    """
    Format ClipMetadata objects for database insertion
    
    Args:
        clips: List of ClipMetadata objects
    
    Returns:
        List of dicts ready for database insertion
    """
    return [
        {
            'keyword': clip.keyword,
            'start_time': clip.start_time,
            'end_time': clip.end_time,
            'duration_seconds': clip.duration_seconds,
            'file_path': clip.file_path,
            'file_size_bytes': clip.file_size_bytes
        }
        for clip in clips
    ]


def format_frames_for_database(frames: List[Dict]) -> List[Dict]:
    """
    Format frame metadata for database insertion
    
    Args:
        frames: List of frame metadata dicts
    
    Returns:
        List of dicts ready for database insertion
    """
    return [
        {
            'timestamp': frame['timestamp'],
            'file_path': frame['file_path'],
            'width': frame['width'],
            'height': frame['height'],
            'contains_chart': frame['contains_chart']
        }
        for frame in frames
    ]