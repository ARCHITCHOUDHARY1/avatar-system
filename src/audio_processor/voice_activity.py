
import numpy as np
import librosa
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    
    def __init__(self, sr: int = 16000, frame_length: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        
    def detect(
        self,
        audio: np.ndarray,
        energy_threshold: float = 0.02,
        zcr_threshold: float = 0.1
    ) -> np.ndarray:
        # Calculate energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # Calculate zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # Voice activity detection
        vad = (energy > energy_threshold) & (zcr < zcr_threshold)
        
        return vad
    
    def get_voiced_segments(
        self,
        audio: np.ndarray,
        min_duration: float = 0.1
    ) -> List[Tuple[int, int]]:
 
        vad = self.detect(audio)
        
        # Convert to sample indices
        min_frames = int(min_duration * self.sr / self.hop_length)
        
        segments = []
        start = None
        
        for i, is_voiced in enumerate(vad):
            if is_voiced and start is None:
                start = i
            elif not is_voiced and start is not None:
                if i - start >= min_frames:
                    segments.append((
                        start * self.hop_length,
                        i * self.hop_length
                    ))
                start = None
        
        # Handle last segment
        if start is not None and len(vad) - start >= min_frames:
            segments.append((
                start * self.hop_length,
                len(audio)
            ))
        
        logger.info(f"Detected {len(segments)} voiced segments")
        return segments
