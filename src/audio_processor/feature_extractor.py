
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sample_rate = self.config.get("audio_sample_rate", 16000)
        self.n_mfcc = self.config.get("n_mfcc", 13)
        self.n_mels = self.config.get("n_mels", 80)
        self.hop_length = self.config.get("hop_length", 512)
        
        logger.info(f"AudioFeatureExtractor initialized: sr={self.sample_rate}, "
                   f"n_mfcc={self.n_mfcc}, n_mels={self.n_mels}")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        try:
            audio_path = Path(audio_path)
            
            # Validate file exists
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Validate file extension
            valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            if audio_path.suffix.lower() not in valid_extensions:
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
            
            logger.debug(f"Loading audio: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            
            # Validate audio
            if len(audio) == 0:
                raise ValueError("Loaded audio is empty")
            
            if np.isnan(audio).any() or np.isinf(audio).any():
                logger.warning("Audio contains NaN or Inf values, cleaning...")
                audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            logger.info(f"Loaded audio: duration={len(audio)/sr:.2f}s, samples={len(audio)}")
            
            return audio
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading audio: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load audio: {e}")
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        try:
            if audio is None or len(audio) == 0:
                raise ValueError("Cannot extract features from empty audio")
            
            logger.debug("Extracting audio features...")
            
            features = {}
            
            # Extract MFCC
            try:
                mfcc = librosa.feature.mfcc(
                    y=audio, 
                    sr=self.sample_rate, 
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length
                )
                features["mfcc"] = mfcc
                logger.debug(f"MFCC shape: {mfcc.shape}")
            except Exception as e:
                logger.error(f"MFCC extraction failed: {e}")
                features["mfcc"] = np.zeros((self.n_mfcc, 1))
            
            # Extract mel spectrogram
            try:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=self.sample_rate, 
                    n_mels=self.n_mels,
                    hop_length=self.hop_length
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                features["mel_spectrogram"] = mel_spec_db
                logger.debug(f"Mel spectrogram shape: {mel_spec_db.shape}")
            except Exception as e:
                logger.error(f"Mel spectrogram extraction failed: {e}")
                features["mel_spectrogram"] = np.zeros((self.n_mels, 1))
            
            # Extract prosody features
            try:
                prosody = self._extract_prosody(audio)
                features["prosody"] = prosody
            except Exception as e:
                logger.error(f"Prosody extraction failed: {e}")
                features["prosody"] = {}
            
            # Extract energy
            try:
                energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
                features["energy"] = energy
            except Exception as e:
                logger.error(f"Energy extraction failed: {e}")
                features["energy"] = np.zeros(1)
            
            # Frame timestamps
            num_frames = features["mfcc"].shape[1]
            features["timestamps"] = np.arange(num_frames) * self.hop_length / self.sample_rate
            
            logger.info(f"Feature extraction complete: {len(features)} feature types, "
                       f"{num_frames} frames")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract features: {e}")
    
    def _extract_prosody(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        try:
            # Extract pitch using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Handle NaN values in pitch
            f0 = np.nan_to_num(f0, nan=0.0)
            
            return {
                "pitch": f0,
                "voiced_flag": voiced_flag,
                "voiced_probability": voiced_probs
            }
        except Exception as e:
            logger.warning(f"Prosody extraction failed: {e}")
            return {
                "pitch": np.zeros(1),
                "voiced_flag": np.zeros(1, dtype=bool),
                "voiced_probability": np.zeros(1)
            }
    
    def extract_chunk_features(self, audio_chunk: List[float]) -> Dict[str, Any]:
        try:
            audio_array = np.array(audio_chunk, dtype=np.float32)
            
            if len(audio_array) < self.hop_length:
                logger.warning(f"Audio chunk too small ({len(audio_array)} samples), padding...")
                audio_array = np.pad(audio_array, (0, self.hop_length - len(audio_array)))
            
            return self.extract_features(audio_array)
            
        except Exception as e:
            logger.error(f"Chunk feature extraction failed: {e}")
            raise
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        try:
            required_keys = ["mfcc", "mel_spectrogram", "timestamps"]
            
            for key in required_keys:
                if key not in features:
                    logger.error(f"Missing required feature: {key}")
                    return False
                
                if isinstance(features[key], np.ndarray):
                    if features[key].size == 0:
                        logger.error(f"Feature {key} is empty")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
