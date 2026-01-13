"""Audio Streamer"""

import asyncio
import numpy as np
from typing import AsyncIterator
import logging

logger = logging.getLogger(__name__)


class AudioStreamer:
    """Stream audio in chunks"""
    
    def __init__(
        self,
        chunk_duration: float = 0.5,
        sample_rate: int = 16000
    ):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        
    async def stream_audio(
        self,
        audio: np.ndarray
    ) -> AsyncIterator[np.ndarray]:
        """Stream audio in chunks"""
        
        num_chunks = len(audio) // self.chunk_size
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = audio[start:end]
            
            yield chunk
            await asyncio.sleep(self.chunk_duration)
        
        # Yield remaining audio
        if len(audio) % self.chunk_size != 0:
            yield audio[num_chunks * self.chunk_size:]
    
    async def stream_from_file(
        self,
        audio_path: str
    ) -> AsyncIterator[np.ndarray]:
        """Stream audio from file"""
        
        import soundfile as sf
        
        with sf.SoundFile(audio_path) as f:
            while True:
                chunk = f.read(self.chunk_size)
                if len(chunk) == 0:
                    break
                
                yield chunk
                await asyncio.sleep(self.chunk_duration)
