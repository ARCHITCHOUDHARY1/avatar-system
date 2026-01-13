"""
Silero VAD - Voice Activity Detection
Ultra-fast speech detection (3ms latency)
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class SileroVAD:
    """Silero Voice Activity Detection"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize Silero VAD
        
        Args:
            threshold: Speech probability threshold (0.0-1.0)
        """
        self.threshold = threshold
        self.model = None
        self.utils = None
        
        logger.info("Loading Silero VAD...")
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model from torch hub"""
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = self.utils
            
            logger.info("[OK] Silero VAD loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise
    
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Quick speech detection for single chunk
        
        Args:
            audio: Audio array (numpy)
            sample_rate: Audio sample rate
            
        Returns:
            True if speech detected
        """
        try:
            # Convert to torch tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            
            # Get speech probability
            speech_prob = self.model(audio, sample_rate).item()
            
            return speech_prob >= self.threshold
            
        except Exception as e:
            logger.error(f"Speech detection failed: {e}")
            return False
    
    def detect_speech_segments(
        self, 
        audio_path: str, 
        return_seconds: bool = True
    ) -> List[dict]:
        """
        Detect all speech segments in audio file
        
        Args:
            audio_path: Path to audio file
            return_seconds: Return timestamps in seconds (vs. samples)
            
        Returns:
            List of speech segments: [{"start": float, "end": float}, ...]
        """
        try:
            # Read audio
            wav = self.read_audio(audio_path, sampling_rate=16000)
            
            # Get timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav, 
                self.model,
                threshold=self.threshold,
                sampling_rate=16000,
                return_seconds=return_seconds
            )
            
            logger.info(f"Detected {len(speech_timestamps)} speech segments")
            return speech_timestamps
            
        except Exception as e:
            logger.error(f"Speech segment detection failed: {e}")
            return []
    
    def filter_silence(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250
    ) -> np.ndarray:
        """
        Remove silence from audio, keep only speech parts
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            min_speech_duration_ms: Minimum speech duration to keep
            
        Returns:
            Filtered audio with silence removed
        """
        try:
            # Convert to torch
            if isinstance(audio, np.ndarray):
                wav = torch.from_numpy(audio).float()
            else:
                wav = audio
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                threshold=self.threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms
            )
            
            # Collect speech chunks
            speech_audio = self.collect_chunks(speech_timestamps, wav)
            
            # Convert back to numpy
            if isinstance(audio, np.ndarray):
                speech_audio = speech_audio.numpy()
            
            reduction = (1 - len(speech_audio) / len(audio)) * 100
            logger.info(f"Filtered {reduction:.1f}% silence")
            
            return speech_audio
            
        except Exception as e:
            logger.error(f"Silence filtering failed: {e}")
            return audio


# Convenience instance
def get_vad(threshold: float = 0.5) -> SileroVAD:
    """Get VAD instance (singleton pattern)"""
    if not hasattr(get_vad, 'instance'):
        get_vad.instance = SileroVAD(threshold=threshold)
    return get_vad.instance
