
import numpy as np
from typing import Iterator, List
import logging

logger = logging.getLogger(__name__)


class ChunkManager:
    
    def __init__(self, chunk_duration: float = 0.5, overlap: float = 0.1, sr: int = 16000):
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sr = sr
        self.chunk_size = int(chunk_duration * sr)
        self.overlap_size = int(overlap * sr)
        
    def create_chunks(self, audio: np.ndarray) -> Iterator[np.ndarray]:
  
        hop_size = self.chunk_size - self.overlap_size
        
        for i in range(0, len(audio), hop_size):
            chunk = audio[i:i + self.chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            
            yield chunk
    
    def reassemble_chunks(
        self,
        chunks: List[np.ndarray],
        original_length: int = None
    ) -> np.ndarray:
        if not chunks:
            return np.array([])
        
        hop_size = self.chunk_size - self.overlap_size
        output_length = (len(chunks) - 1) * hop_size + self.chunk_size
        output = np.zeros(output_length)
        
        for i, chunk in enumerate(chunks):
            start = i * hop_size
            end = start + self.chunk_size
            
            if i == 0:
                # First chunk - no fade in
                output[start:end] = chunk
            elif i == len(chunks) - 1:
                # Last chunk - crossfade and trim
                output[start:start + self.overlap_size] = (
                    output[start:start + self.overlap_size] * np.linspace(1, 0, self.overlap_size) +
                    chunk[:self.overlap_size] * np.linspace(0, 1, self.overlap_size)
                )
                output[start + self.overlap_size:end] = chunk[self.overlap_size:]
            else:
                # Middle chunks - crossfade overlap
                output[start:start + self.overlap_size] = (
                    output[start:start + self.overlap_size] * np.linspace(1, 0, self.overlap_size) +
                    chunk[:self.overlap_size] * np.linspace(0, 1, self.overlap_size)
                )
                output[start + self.overlap_size:end] = chunk[self.overlap_size:]
        
        # Trim to original length if provided
        if original_length:
            output = output[:original_length]
        
        return output
