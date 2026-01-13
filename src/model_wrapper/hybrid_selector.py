"""
Hybrid Model Selector - Auto-selects best model based on available hardware

Automatically uses:
- Wav2Lip: When running on CPU (fast, 5 min per video)
- SadTalker: When GPU is available (best quality, 60 sec per video)

Production-ready with comprehensive error handling and validation.
"""

import torch
import logging
import os
from typing import Literal, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class HybridModelSelector:
    """Automatically select and load the best model based on hardware"""
    
    def __init__(self, force_model: Optional[str] = None):
        """
        Args:
            force_model: Force specific model ('wav2lip' or 'sadtalker')
                        If None, auto-detect based on GPU availability
        """
        try:
            self.device = self._detect_device()
            self.model_type = self._select_model(force_model)
            self.model = None
            self.model_loaded = False
            
            logger.info(f"[HYBRID] Selected: {self.model_type.upper()} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid selector: {e}")
            # Fallback to safest option
            self.device = "cpu"
            self.model_type = "wav2lip"
            self.model = None
            self.model_loaded = False
            logger.warning(f"Fallback to CPU+Wav2Lip due to error")
    
    def _detect_device(self) -> str:
        """Detect if GPU is available with error handling"""
        try:
            if torch.cuda.is_available():
                # Verify GPU is actually accessible
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    # Test if we can allocate memory
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    logger.info(f"[OK] GPU detected and verified: {gpu_name}")
                    return "cuda"
                except Exception as e:
                    logger.warning(f"GPU detected but not accessible: {e}")
                    logger.info("Falling back to CPU")
                    return "cpu"
            else:
                logger.info("[INFO] No GPU detected, using CPU")
                return "cpu"
        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            logger.info("Defaulting to CPU")
            return "cpu"
    
    def _select_model(self, force_model: Optional[str] = None) -> Literal["wav2lip", "sadtalker"]:
        """
        Select model based on hardware with validation
        
        Logic:
        - GPU available → SadTalker (best quality)
        - CPU only → Wav2Lip (optimized for CPU)
        """
        if force_model:
            # Validate forced model
            force_model = force_model.lower().strip()
            if force_model not in ["wav2lip", "sadtalker"]:
                logger.warning(f"Invalid force_model '{force_model}'. Must be 'wav2lip' or 'sadtalker'")
                logger.info("Auto-detecting instead...")
                force_model = None
            else:
                logger.info(f"[FORCE] Using forced model: {force_model}")
                return force_model
        
        if self.device == "cuda":
            logger.info("[AUTO] GPU detected → Using SadTalker (high quality)")
            return "sadtalker"
        else:
            logger.info("[AUTO] CPU detected → Using Wav2Lip (CPU-optimized)")
            return "wav2lip"
    
    def load_model(self):
        """Load the selected model with comprehensive error handling"""
        if self.model_loaded:
            logger.info(f"{self.model_type.upper()} already loaded")
            return self.model
        
        logger.info(f"Loading {self.model_type.upper()} model...")
        
        try:
            if self.model_type == "wav2lip":
                self._load_wav2lip()
            else:
                self._load_sadtalker()
                
            self.model_loaded = True
            logger.info(f"[OK] {self.model_type.upper()} loaded successfully")
            return self.model
            
        except ImportError as e:
            logger.error(f"Import error while loading {self.model_type}: {e}")
            logger.info("Make sure all dependencies are installed: pip install -r requirements.txt")
            raise RuntimeError(f"Failed to import {self.model_type} dependencies") from e
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            if self.model_type == "wav2lip":
                logger.info("Download Wav2Lip model:")
                logger.info("wget https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth -P models/wav2lip/checkpoints/")
            else:
                logger.info("Download SadTalker checkpoints:")
                logger.info("python download_sadtalker_checkpoints.py")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error loading {self.model_type}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load {self.model_type}") from e
    
    def _load_wav2lip(self):
        """Load Wav2Lip model with safety checks"""
        try:
            from models.wav2lip_model import Wav2LipModel
        except ImportError:
            logger.error("Cannot import Wav2LipModel. Check if models/wav2lip_model.py exists")
            raise
        
        self.model = Wav2LipModel(device=self.device)
        logger.debug("Wav2LipModel instance created")
    
    def _load_sadtalker(self):
        """Load SadTalker model with safety checks"""
        try:
            from models.sadtalker_integration import SadTalkerModel
        except ImportError:
            logger.error("Cannot import SadTalkerModel. Check if models/sadtalker_integration.py exists")
            raise
        
        self.model = SadTalkerModel(device=self.device)
        logger.debug("SadTalkerModel instance created")
    
    def _validate_inputs(self, image_path: str, audio_path: str, output_path: str):
        """Validate input/output paths"""
        # Check image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check audio exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check image is readable
        image_ext = Path(image_path).suffix.lower()
        valid_image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
        if image_ext not in valid_image_exts:
            raise ValueError(f"Invalid image format '{image_ext}'. Must be one of: {valid_image_exts}")
        
        # Check audio is readable
        audio_ext = Path(audio_path).suffix.lower()
        valid_audio_exts = ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
        if audio_ext not in valid_audio_exts:
            raise ValueError(f"Invalid audio format '{audio_ext}'. Must be one of: {valid_audio_exts}")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check output format
        output_ext = Path(output_path).suffix.lower()
        if output_ext not in ['.mp4', '.avi', '.mov']:
            logger.warning(f"Output format '{output_ext}' may not be supported. Recommended: .mp4")
    
    def generate(
        self, 
        image_path: str, 
        audio_path: str, 
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate talking avatar video with comprehensive validation
        
        Args:
            image_path: Path to source image
            audio_path: Path to audio file
            output_path: Path to save output video
            **kwargs: Additional parameters (fps, resolution, etc.)
            
        Returns:
            Path to generated video
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If input formats are invalid
            RuntimeError: If generation fails
        """
        try:
            # Validate inputs
            self._validate_inputs(image_path, audio_path, output_path)
            
            # Load model if not already loaded
            if not self.model_loaded:
                self.load_model()
            
            logger.info(f"[{self.model_type.upper()}] Generating avatar...")
            logger.debug(f"Image: {image_path}")
            logger.debug(f"Audio: {audio_path}")
            logger.debug(f"Output: {output_path}")
            
            # Generate with both models having same interface
            result = self.model.generate(
                image_path=image_path,
                audio_path=audio_path,
                output_path=output_path,
                **kwargs
            )
            
            # Verify output was created
            if not os.path.exists(result):
                raise RuntimeError(f"Generation completed but output file not found: {result}")
            
            # Check output file size
            output_size = os.path.getsize(result)
            if output_size < 1024:  # Less than 1KB is suspicious
                logger.warning(f"Output file is very small ({output_size} bytes). Generation may have failed.")
            
            logger.info(f"[OK] Generation complete: {result} ({output_size / 1024:.1f} KB)")
            return result
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Avatar generation failed") from e
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about selected model and hardware"""
        info = {
            "model": self.model_type,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "gpu_available": False,
        }
        
        try:
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info["gpu_count"] = torch.cuda.device_count()
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
        
        return info


# Convenience function
def create_hybrid_generator(force_model: Optional[str] = None):
    """
    Create and return hybrid model generator with error handling
    
    Usage:
        generator = create_hybrid_generator()
        generator.generate("face.jpg", "audio.wav", "output.mp4")
    
    Args:
        force_model: Optional, force 'wav2lip' or 'sadtalker'
        
    Returns:
        HybridModelSelector instance
    """
    try:
        return HybridModelSelector(force_model=force_model)
    except Exception as e:
        logger.error(f"Failed to create hybrid generator: {e}")
        logger.warning("Falling back to CPU-only Wav2Lip mode")
        return HybridModelSelector(force_model="wav2lip")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    try:
        selector = HybridModelSelector()
        info = selector.get_info()
        
        print("\n" + "="*60)
        print("HYBRID MODEL SELECTOR")
        print("="*60)
        print(f"Selected Model: {info['model'].upper()}")
        print(f"Device: {info['device'].upper()}")
        print(f"GPU Available: {info['gpu_available']}")
        if info['gpu_available']:
            print(f"GPU Name: {info['gpu_name']}")
            print(f"GPU Memory: {info['gpu_memory_gb']:.1f} GB")
            print(f"GPU Count: {info['gpu_count']}")
        print("="*60)
        
        print("\nRecommendations:")
        if info['model'] == 'wav2lip':
            print("- Using Wav2Lip (CPU-optimized)")
            print("- Expected time: 3-5 minutes per 10-second video")
            print("- Recommended resolution: 256x256")
        else:
            print("- Using SadTalker (GPU-accelerated)")
            print("- Expected time: 30-60 seconds per 10-second video")
            print("- Recommended resolution: 512x512")
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
