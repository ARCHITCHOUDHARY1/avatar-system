"""
Hybrid Video Generation Node
Automatically selects Wav2Lip (CPU) or SadTalker (GPU) based on hardware
"""

from typing import Dict, Any
import logging
import time
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class HybridVideoGenerator:
    """
    Intelligent model selection:
    - Wav2Lip for CPU-only systems (faster, optimized)
    - SadTalker for GPU systems (higher quality)
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        logger.info("Initializing Hybrid Video Generator...")
    
    def generate(self, image_path, audio_path, output_path, fps=25, resolution=(512, 512), emotion=None):
        """Generate video with automatic model selection"""
        
        # Detect hardware
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            model_type = "sadtalker"
            logger.info("🎮 GPU detected - Using SadTalker for high quality")
        else:
            model_type = "wav2lip"
            logger.info("💻 CPU only - Using Wav2Lip for optimized performance")
        
        # Load model if needed
        if self.model is None or self.model_type != model_type:
            self._load_model(model_type)
        
        # Generate video
        if model_type == "sadtalker":
            return self._generate_sadtalker(image_path, audio_path, output_path, emotion)
        else:
            return self._generate_wav2lip(image_path, audio_path, output_path, fps, resolution)
    
    def _load_model(self, model_type):
        """Load specified model with fallback"""
        try:
            if model_type == "sadtalker":
                logger.info("Loading SadTalker...")
                from models.sadtalker_integration import EnhancedSadTalker
                self.model = EnhancedSadTalker()
                self.model.load_model()
                self.model_type = "sadtalker"
                logger.info("✅ SadTalker loaded")
            else:
                logger.info("Loading Wav2Lip...")
                from models.wav2lip_model import Wav2LipModel
                self.model = Wav2LipModel()
                self.model.load_model()
                self.model_type = "wav2lip"
                logger.info("✅ Wav2Lip loaded")
                
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            
            # Fallback to Wav2Lip
            if model_type == "sadtalker":
                logger.warning("⚠️ Falling back to Wav2Lip...")
                from models.wav2lip_model import Wav2LipModel
                self.model = Wav2LipModel()
                self.model.load_model()
                self.model_type = "wav2lip"
                logger.info("✅ Wav2Lip fallback successful")
            else:
                raise
    
    def _generate_sadtalker(self, image_path, audio_path, output_path, emotion):
        """Generate with SadTalker"""
        return self.model.generate_video(
            source_image=image_path,
            driven_audio=audio_path,
            output_path=output_path,
            emotion_hint=emotion
        )
    
    def _generate_wav2lip(self, image_path, audio_path, output_path, fps, resolution):
        """Generate with Wav2Lip"""
        return self.model.generate(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
            fps=fps,
            resolution=resolution[0] if isinstance(resolution, (tuple, list)) else resolution
        )
