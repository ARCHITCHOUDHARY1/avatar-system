"""
Multiple TTS Implementations - CPU/GPU Compatible
Includes: gTTS (Google), Edge-TTS (Microsoft), fallback to pyttsx3
"""

import torch
import numpy as np
import logging
from typing import Optional, Union
from pathlib import Path
import soundfile as sf
import asyncio

logger = logging.getLogger(__name__)


class MultiTTS:
    """
    Multi-backend TTS with automatic fallback
    Backends: gTTS (Google), edge-tts (Microsoft), pyttsx3 (offline)
    """
    
    def __init__(
        self, 
        backend: str = "edge",  # "gtts", "edge", or "pyttsx3"
        device: Optional[str] = None
    ):
        """
        Initialize TTS
        
        Args:
            backend: TTS backend ("gtts", "edge", "pyttsx3")
            device: Device (CPU/CUDA) - only affects some backends
        """
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 24000
        
        logger.info(f"Initializing {backend} TTS on {self.device}...")
        self._load_model()
    
    def _load_model(self):
        """Load TTS model based on backend"""
        try:
            if self.backend == "gtts":
                from gtts import gTTS
                self.model = gTTS
                logger.info("[OK] gTTS loaded successfully")
                
            elif self.backend == "edge":
                import edge_tts
                self.model = edge_tts
                logger.info("[OK] Edge-TTS loaded successfully")
                
            elif self.backend == "pyttsx3":
                import pyttsx3
                self.model = pyttsx3.init()
                logger.info("[OK] pyttsx3 loaded successfully")
                
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Failed to load {self.backend}: {e}")
            self._load_fallback()
    
    def _load_fallback(self):
        """Fallback to simplest TTS"""
        try:
            logger.warning("Loading fallback TTS (pyttsx3)...")
            import pyttsx3
            self.backend = "pyttsx3"
            self.model = pyttsx3.init()
            logger.info("[OK] Fallback TTS loaded")
        except Exception as e:
            logger.error(f"All TTS backends failed: {e}")
            raise RuntimeError("No TTS available")
    
    def synthesize(
        self, 
        text: str, 
        output_path: Optional[str] = None,
        voice: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            output_path: Optional path to save audio
            voice: Optional voice ID (backend-specific)
            
        Returns:
            Audio array (numpy)
        """
        try:
            logger.info(f"Synthesizing ({self.backend}): '{text[:50]}...'")
            
            if self.backend == "gtts":
                return self._synthesize_gtts(text, output_path)
            elif self.backend == "edge":
                return self._synthesize_edge(text, output_path, voice)
            elif self.backend == "pyttsx3":
                return self._synthesize_pyttsx3(text, output_path)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def _synthesize_gtts(self, text: str, output_path: Optional[str]) -> np.ndarray:
        """Synthesize using Google TTS"""
        import tempfile
        from pydub import AudioSegment
        
        # Create temp file if no output path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Generate speech
        tts = self.model(text=text, lang='en', slow=False)
        tts.save(output_path)
        
        # Load and convert to numpy
        audio = AudioSegment.from_mp3(output_path)
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)  # Mono
        
        # Convert to numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize
        
        logger.info(f"[OK] Synthesized {len(samples)/self.sample_rate:.2f}s")
        return samples
    
    def _synthesize_edge(self, text: str, output_path: Optional[str], voice: Optional[str]) -> np.ndarray:
        """Synthesize using Microsoft Edge TTS"""
        import tempfile
        from pydub import AudioSegment
        
        # Default voice
        if voice is None:
            voice = "en-US-AriaNeural"  # Female voice
            # voice = "en-US-GuyNeural"  # Male voice
        
        # Create temp file if needed
        temp_output = output_path or tempfile.mktemp(suffix=".mp3")
        
        # Generate speech (async)
        async def generate():
            communicate = self.model.Communicate(text, voice)
            await communicate.save(temp_output)
        
        # Run async
        asyncio.run(generate())
        
        # Load and convert
        audio = AudioSegment.from_mp3(temp_output)
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        
        logger.info(f"[OK] Synthesized {len(samples)/self.sample_rate:.2f}s")
        return samples
    
    def _synthesize_pyttsx3(self, text: str, output_path: Optional[str]) -> np.ndarray:
        """Synthesize using pyttsx3 (offline)"""
        import tempfile
        from pydub import AudioSegment
        
        # Create temp file
        temp_output = output_path or tempfile.mktemp(suffix=".wav")
        
        # Generate
        self.model.save_to_file(text, temp_output)
        self.model.runAndWait()
        
        # Load
        audio, sr = sf.read(temp_output)
        
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        logger.info(f"[OK] Synthesized {len(audio)/self.sample_rate:.2f}s")
        return audio
    
    def batch_synthesize(self, texts: list, output_dir: Optional[str] = None) -> list:
        """Batch synthesize multiple texts"""
        outputs = []
        
        for i, text in enumerate(texts):
            output_path = None
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_path = f"{output_dir}/speech_{i:03d}.mp3"
            
            audio = self.synthesize(text, output_path)
            outputs.append(audio)
        
        logger.info(f"Batch synthesized {len(texts)} utterances")
        return outputs


# Convenience function
def get_tts(backend: str = "edge", device: Optional[str] = None) -> MultiTTS:
    """Get TTS instance (singleton pattern)"""
    cache_key = f"{backend}_{device}"
    
    if not hasattr(get_tts, 'instances'):
        get_tts.instances = {}
    
    if cache_key not in get_tts.instances:
        get_tts.instances[cache_key] = MultiTTS(backend, device=device)
    
    return get_tts.instances[cache_key]


# Backward compatibility
ChatterboxTTS = MultiTTS

