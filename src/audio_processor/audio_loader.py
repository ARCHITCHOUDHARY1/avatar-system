
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        
    def load(
        self,
        audio_path: str,
        normalize: bool = True,
        trim_silence: bool = True
    ) -> Tuple[np.ndarray, int]:

        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Loading audio: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
            
            # Trim silence
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Normalize
            if normalize:
                audio = librosa.util.normalize(audio)
            
            logger.info(f"Loaded audio: duration={len(audio)/sr:.2f}s, sr={sr}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def save(
        self,
        audio: np.ndarray,
        output_path: str,
        sr: int = None
    ) -> None:
        sr = sr or self.target_sr
        sf.write(output_path, audio, sr)
        logger.info(f"Saved audio to: {output_path}")
    
    def get_duration(self, audio: np.ndarray, sr: int) -> float:
        return len(audio) / sr
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
