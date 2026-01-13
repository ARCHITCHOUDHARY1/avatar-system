
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FrameGenerator:
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.resolution = tuple(self.config.get("video_resolution", (512, 512)))
        self.fps = self.config.get("video_fps", 30)
        
        logger.info(f"FrameGenerator initialized: resolution={self.resolution}, fps={self.fps}")
    
    def generate_frames(
        self,
        face_parameters: Dict[str, np.ndarray],
        source_image: np.ndarray
    ) -> List[np.ndarray]:

        try:
            if source_image is None or source_image.size == 0:
                raise ValueError("Source image is empty")
            
            if not face_parameters:
                raise ValueError("Face parameters are empty")
            
            logger.debug("Generating video frames...")
            
            # Validate and resize source image
            source_image = self._prepare_source_image(source_image)
            
            frames = []
            num_frames = self._get_num_frames(face_parameters)
            
            logger.info(f"Generating {num_frames} frames...")
            
            for frame_idx in range(num_frames):
                try:
                    # Generate frame
                    frame = self._generate_single_frame(
                        source_image,
                        face_parameters,
                        frame_idx
                    )
                    
                    # Validate frame
                    if self._validate_frame(frame):
                        frames.append(frame)
                    else:
                        logger.warning(f"Invalid frame at index {frame_idx}, using previous frame")
                        if frames:
                            frames.append(frames[-1].copy())
                        else:
                            frames.append(source_image.copy())
                    
                except Exception as e:
                    logger.error(f"Frame generation failed at index {frame_idx}: {e}")
                    # Use previous frame or source as fallback
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(source_image.copy())
            
            logger.info(f"Successfully generated {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Frame generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Frame generation error: {e}")
    
    def _prepare_source_image(self, image: np.ndarray) -> np.ndarray:
        try:
            # Ensure RGB format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Resize to target resolution
            if image.shape[:2] != self.resolution:
                image = cv2.resize(image, self.resolution)
            
            # Validate pixel values
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 255)).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            raise
    
    def _get_num_frames(self, face_parameters: Dict[str, np.ndarray]) -> int:
        try:
            # Get frame count from parameters
            for key in ['pose', 'expression', 'coefficients']:
                if key in face_parameters:
                    param = face_parameters[key]
                    if isinstance(param, np.ndarray):
                        return len(param) if param.ndim > 1 else 1
            
            # Default to 1 frame
            return 1
            
        except Exception as e:
            logger.warning(f"Could not determine frame count: {e}, using 1")
            return 1
    
    def _generate_single_frame(
        self,
        source_image: np.ndarray,
        face_parameters: Dict[str, np.ndarray],
        frame_idx: int
    ) -> np.ndarray:
        try:
            # TODO: Implement actual frame rendering with SadTalker
            # For now, return source image with minor variations
            
            frame = source_image.copy()
            
            # Apply slight variations based on frame index (placeholder)
            if 'pose' in face_parameters:
                # Apply pose transformation (placeholder)
                pass
            
            if 'expression' in face_parameters:
                # Apply expression (placeholder)
                pass
            
            return frame
            
        except Exception as e:
            logger.error(f"Single frame generation failed: {e}")
            return source_image.copy()
    
    def _validate_frame(self, frame: np.ndarray) -> bool:
        try:
            if frame is None:
                return False
            
            if frame.size == 0:
                return False
            
            if frame.shape[:2] != self.resolution:
                return False
            
            if np.isnan(frame).any() or np.isinf(frame).any():
                return False
            
            return True
            
        except Exception:
            return False
    
    def apply_filter(self, frame: np.ndarray, filter_type: str = "none") -> np.ndarray:
        try:
            if filter_type == "none":
                return frame
            elif filter_type == "blur":
                return cv2.GaussianBlur(frame, (5, 5), 0)
            elif filter_type == "sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                return cv2.filter2D(frame, -1, kernel)
            elif filter_type == "grayscale":
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                logger.warning(f"Unknown filter type: {filter_type}")
                return frame
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return frame
