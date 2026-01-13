"""Video Streamer"""

import asyncio
import numpy as np
import cv2
from typing import AsyncIterator, List
import logging

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Stream video frames"""
    
    def __init__(self, fps: int = 25):
        self.fps = fps
        self.frame_duration = 1.0 / fps
        
    async def stream_frames(
        self,
        frames: List[np.ndarray]
    ) -> AsyncIterator[np.ndarray]:
        """Stream video frames"""
        
        for frame in frames:
            yield frame
            await asyncio.sleep(self.frame_duration)
    
    async def stream_encoded_frames(
        self,
        frames: List[np.ndarray],
        quality: int = 90
    ) -> AsyncIterator[bytes]:
        """Stream encoded frames as JPEG"""
        
        for frame in frames:
            # Encode frame as JPEG
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            
            yield buffer.tobytes()
            await asyncio.sleep(self.frame_duration)
    
    async def stream_from_generator(
        self,
        frame_generator: callable
    ) -> AsyncIterator[np.ndarray]:
        """Stream frames from generator function"""
        
        while True:
            try:
                frame = next(frame_generator)
                yield frame
                await asyncio.sleep(self.frame_duration)
            except StopIteration:
                break
