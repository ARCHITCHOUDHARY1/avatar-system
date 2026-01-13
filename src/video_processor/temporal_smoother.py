"""Temporal Smoother - Enhanced with multiple smoothing methods"""

import numpy as np
import cv2
from typing import List
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Apply temporal smoothing to video frames with error handling"""
    
    def __init__(self, window_size: int = 5, sigma: float = 1.0):
        self.window_size = window_size
        self.sigma = sigma
        
        logger.info(f"TemporalSmoother initialized: window={window_size}, sigma={sigma}")
    
    def smooth_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal smoothing to frame sequence
        
        Args:
            frames: List of video frames
        
        Returns:
            Smoothed frames
        """
        try:
            if not frames:
                logger.warning("No frames to smooth")
                return frames
            
            if len(frames) < self.window_size:
                logger.warning(f"Frame count ({len(frames)}) less than window size "
                             f"({self.window_size}), using smaller window")
                window = max(1, len(frames) // 2)
            else:
                window = self.window_size
            
            logger.debug(f"Smoothing {len(frames)} frames with window={window}")
            
            # Convert to numpy array
            try:
                frames_array = np.array(frames, dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to convert frames to array: {e}")
                return frames
            
            # Validate frames
            if frames_array.ndim != 4:
                logger.error(f"Invalid frame array shape: {frames_array.shape}")
                return frames
            
            # Apply gaussian smoothing along time axis
            try:
                smoothed = gaussian_filter1d(
                    frames_array,
                    sigma=self.sigma,
                    axis=0,
                    mode='nearest'
                )
                
                # Convert back to uint8
                smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
                
                # Convert back to list
                smoothed_frames = [frame for frame in smoothed]
                
                logger.info(f"Temporal smoothing complete: {len(smoothed_frames)} frames")
                
                return smoothed_frames
                
            except Exception as e:
                logger.error(f"Gaussian smoothing failed: {e}")
                return frames
            
        except Exception as e:
            logger.error(f"Frame smoothing failed: {e}", exc_info=True)
            return frames
    
    def smooth_with_moving_average(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply moving average smoothing"""
        try:
            if len(frames) < 2:
                return frames
            
            smoothed = []
            half_window = self.window_size // 2
            
            for i in range(len(frames)):
                start = max(0, i - half_window)
                end = min(len(frames), i + half_window + 1)
                
                window_frames = frames[start:end]
                avg_frame = np.mean(window_frames, axis=0, dtype=np.float32)
                avg_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)
                
                smoothed.append(avg_frame)
            
            logger.info("Moving average smoothing complete")
            return smoothed
            
        except Exception as e:
            logger.error(f"Moving average smoothing failed: {e}")
            return frames
    
    def smooth_optical_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Smooth frames using optical flow (placeholder)"""
        try:
            if len(frames) < 2:
                return frames
            
            logger.debug("Optical flow smoothing (placeholder)")
            
            # TODO: Implement actual optical flow smoothing
            # For now, return original frames
            return frames
            
        except Exception as e:
            logger.error(f"Optical flow smoothing failed: {e}")
            return frames
    
    def remove_jitter(self, frames: List[np.ndarray], threshold: float = 10.0) -> List[np.ndarray]:
        """Remove jitter by detecting and smoothing large frame differences"""
        try:
            if len(frames) < 2:
                return frames
            
            smoothed = [frames[0]]
            
            for i in range(1, len(frames)):
                # Calculate frame difference
                diff = np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32))
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    # Large difference detected, blend with previous frame
                    logger.debug(f"Jitter detected at frame {i}, diff={mean_diff:.2f}")
                    blended = cv2.addWeighted(frames[i-1], 0.5, frames[i], 0.5, 0)
                    smoothed.append(blended)
                else:
                    smoothed.append(frames[i])
            
            logger.info(f"Jitter removal complete")
            return smoothed
            
        except Exception as e:
            logger.error(f"Jitter removal failed: {e}")
            return frames
