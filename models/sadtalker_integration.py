import sys
from pathlib import Path

# CRITICAL: Apply compatibility patches FIRST, before any other imports
# This must be done before importing torch, torchvision, or SadTalker modules
try:
    # Add models directory to path if not already there
    models_dir = Path(__file__).parent
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    
    # Import and apply patches
    import compatibility_patches
except ImportError as e:
    import logging
    logging.warning(f"Could not import compatibility patches: {e}")

import torch
import numpy as np
import cv2
import logging


logger = logging.getLogger(__name__)


class EnhancedSadTalker:
    """Enhanced SadTalker with Mistral integration for smarter avatar generation"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.audio_processor = None
        self.model_loaded = False
        
        logger.info(f"EnhancedSadTalker initialized on {self.device}")
    
    def load_model(self):
        """Load SadTalker with optimizations for Colab"""
        try:
            logger.info("Loading SadTalker with optimizations...")
            
            # Add SadTalker to path
            sadtalker_path = Path('models/sadtalker')
            if not sadtalker_path.exists():
                raise FileNotFoundError(f"SadTalker not found at {sadtalker_path}")
            
            sys.path.insert(0, str(sadtalker_path))
            
            # Import SadTalker components
            try:
                from src.facerender.animate import AnimateFromCoeff
                from src.test_audio2coeff import Audio2Coeff  
                from src.utils.preprocess import CropAndExtract
                from src.generate_batch import get_data
                from src.generate_facerender_batch import get_facerender_data
                from src.utils.init_path import init_path
            except ImportError as e:
                logger.error(f"Failed to import SadTalker modules: {e}")
                logger.info("Make sure SadTalker is properly installed in models/sadtalker/")
                raise
            
            # Define paths dictionary
            checkpoint_path = Path('models/sadtalker/checkpoints')
            sadtalker_paths = {
                "use_safetensor": True,
                "checkpoint": str(checkpoint_path / "SadTalker_V0.0.2_256.safetensors"),
                "dir_of_BFM_fitting": str(Path("models/sadtalker/src/config")), 
                "audio2pose_yaml_path": str(Path("models/sadtalker/src/config/auido2pose.yaml")),
                "audio2exp_yaml_path": str(Path("models/sadtalker/src/config/auido2exp.yaml")),
                "facerender_yaml": str(Path("models/sadtalker/src/config/facerender.yaml")),
                "mappingnet_checkpoint": str(checkpoint_path / "mapping_00109-model.pth.tar"),
                "facerender_still_yaml": str(Path("models/sadtalker/src/config/facerender_still.yaml")),
                "preprocess": "crop"
            }

            # Check if files exist
            for key, val in sadtalker_paths.items():
                if isinstance(val, str) and (val.endswith(".yaml") or val.endswith(".safetensors") or val.endswith(".tar")):
                    if not Path(val).exists():
                         logger.warning(f"File not found: {val}")

            # Initialize components
            self.preprocess_model = CropAndExtract(sadtalker_paths, self.device)
            self.audio2coeff = Audio2Coeff(sadtalker_paths, self.device)
            self.animate = AnimateFromCoeff(sadtalker_paths, self.device)
            
            # Save helper functions
            self.get_data = get_data
            self.get_facerender_data = get_facerender_data
            
            self.model_loaded = True
            logger.info(f"SadTalker loaded successfully on {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SadTalker: {e}", exc_info=True)
            self.model_loaded = False
            return False
    
    def generate_video(self, source_image, driven_audio, output_path=None, emotion_hint=None):
        """
        Generate video with optional emotion guidance from Mistral
        """
        try:
            if not self.model_loaded:
                logger.warning("Model not loaded, attempting to load...")
                if not self.load_model():
                    raise RuntimeError("Failed to load SadTalker model")
            
            logger.info(f"Generating video: {source_image} + {driven_audio}")
            
            # Prepare output path
            if output_path is None:
                output_dir = Path("data/outputs/videos")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "output.mp4"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
            save_dir = output_path.parent
            tmp_dir = save_dir / "tmp"
            tmp_dir.mkdir(exist_ok=True)

            # 1. Preprocess Image (Crop & Extract 3DMM)
            logger.info("Preprocessing image...")
            first_frame_dir = tmp_dir / "first_frame_dir"
            first_frame_dir.mkdir(exist_ok=True)
            
            first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
                input_path=str(source_image),
                save_dir=str(first_frame_dir),
                crop_or_resize="crop", # Default to crop
                source_image_flag=True,
                pic_size=256
            )
            
            if first_coeff_path is None:
                raise RuntimeError("Failed to preprocess image (no coeffs)")

            # 2. Audio to Coefficients
            logger.info("Generating coefficients from audio...")
            batch = self.get_data(
                first_coeff_path=first_coeff_path,
                audio_path=str(driven_audio),
                device=self.device,
                ref_eyeblink_coeff_path=None,
                still=False # default
            )
            
            pose_style = self._get_pose_style(emotion_hint)
            coeff_path = self.audio2coeff.generate(
                batch=batch,
                coeff_save_dir=str(tmp_dir),
                pose_style=pose_style
            )

            # 3. Animate (Coeffs to Video)
            logger.info("Rendering video...")
            data = self.get_facerender_data(
                coeff_path=coeff_path,
                pic_path=crop_pic_path,
                first_coeff_path=first_coeff_path,
                audio_path=str(driven_audio),
                batch_size=1, # Reduced for stability
                input_yaw_list=None,
                input_pitch_list=None,
                input_roll_list=None,
                expression_scale=1.0,
                still_mode=False,
                preprocess="crop",
                size=256
            )
            
            video_path = self.animate.generate(
                x=data,
                video_save_dir=str(save_dir),
                pic_path=str(source_image),
                crop_info=crop_info,
                enhancer=None, # enhancer handled separately by orchestrator if needed
                background_enhancer=None,
                preprocess="crop",
                img_size=256
            )
            
            # Move/Rename video to requested output_path if needed
            final_path = Path(video_path)
            if final_path != output_path:
                if output_path.exists():
                    output_path.unlink()
                final_path.rename(output_path)
            
            # Cleanup tmp
            # import shutil
            # shutil.rmtree(tmp_dir) # Keep for debug
            
            logger.info(f"Video generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            raise
    
    def generate_realtime(self, audio_chunk, image):
        """Generate frame from audio chunk in real-time"""
        try:
            # Process audio chunk
            audio_features = self.extract_audio_features(audio_chunk)
            
            # Generate face parameters
            face_params = self.predict_face_params(audio_features)
            
            # Render frame
            frame = self.render_frame(image, face_params)
            
            return frame
            
        except Exception as e:
            logger.error(f"Real-time generation failed: {e}")
            return image  # Return original image as fallback
    
    def extract_audio_features(self, audio_chunk):
        """Extract features from audio chunk"""
        try:
            # Convert to numpy if needed
            if isinstance(audio_chunk, list):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)
            
            # Extract features using audio processor
            # This is a placeholder - implement actual feature extraction
            features = {
                'mfcc': np.zeros((13, 100)),
                'energy': np.zeros(100)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def predict_face_params(self, audio_features):
        """Predict face parameters from audio features"""
        try:
            # This is a placeholder - implement actual prediction
            face_params = {
                'pose': np.zeros((6,)),
                'expression': np.zeros((64,))
            }
            
            return face_params
            
        except Exception as e:
            logger.error(f"Face parameter prediction failed: {e}")
            return {}
    
    def render_frame(self, image, face_params):
        """Render frame with face parameters"""
        try:
            # This is a placeholder - implement actual rendering
            # For now, return the original image
            if isinstance(image, str):
                frame = cv2.imread(str(image))
            else:
                frame = image
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame rendering failed: {e}")
            return image
    
    def _get_pose_style(self, emotion_hint):
        """Convert emotion hint to pose style"""
        emotion_map = {
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'surprised': 4,
            'neutral': 0
        }
        
        if isinstance(emotion_hint, str):
            emotion_hint = emotion_hint.lower()
            for key in emotion_map:
                if key in emotion_hint:
                    return emotion_map[key]
        
        return 0  # Default neutral
    
    def optimize_for_colab(self):
        """Apply Colab-specific optimizations"""
        try:
            logger.info("Applying Colab optimizations...")
            
            # Enable mixed precision
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision('medium')
            
            # Enable gradient checkpointing if available
            if self.model and hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Optimizations applied")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'audio2coeff'):
                del self.audio2coeff
            if hasattr(self, 'animate'):
                del self.animate
            if hasattr(self, 'model'):
                del self.model
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SadTalker cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
