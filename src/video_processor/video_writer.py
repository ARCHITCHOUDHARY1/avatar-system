"""Video Writer - Write video files"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoWriter:
    """Handle video file writing"""
    
    def __init__(
        self,
        fps: int = 25,
        codec: str = "mp4v",
        quality: int = 90
    ):
        self.fps = fps
        self.codec = codec
        self.quality = quality
        
    def write(
        self,
        frames: List[np.ndarray],
        output_path: str,
        audio_path: Optional[str] = None
    ) -> None:
        """
        Write frames to video file
        
        Args:
            frames: List of video frames
            output_path: Output video path
            audio_path: Optional audio file to merge
        """
        
        if not frames:
            raise ValueError("No frames to write")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame properties
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (width, height)
        )
        
        # Write frames
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        
        logger.info(f"Wrote {len(frames)} frames to {output_path}")
        
        # Merge audio if provided
        if audio_path:
            self._merge_audio(output_path, audio_path)
    
    def _merge_audio(self, video_path: Path, audio_path: str) -> None:
        """Merge audio with video using ffmpeg"""
        
        import subprocess
        
        output_with_audio = video_path.parent / f"{video_path.stem}_final.mp4"
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            str(output_with_audio)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Merged audio to video: {output_with_audio}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to merge audio: {e}")
    
    def write_frames_individually(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame"
    ) -> None:
        """Write frames as individual images"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            output_path = output_dir / f"{prefix}_{i:06d}.png"
            cv2.imwrite(str(output_path), frame)
        
        logger.info(f"Wrote {len(frames)} frames to {output_dir}")
