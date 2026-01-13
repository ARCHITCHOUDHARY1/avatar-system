
import asyncio
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import uuid

from .graph_builder import MistralAvatarOrchestrator
from .state_manager import AvatarState
from ..monitor.performance_monitor import PerformanceMonitor
from ..observability import LangfuseMonitor, is_langfuse_enabled

import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class PipelineRunner:
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.orchestrator = MistralAvatarOrchestrator()
        self.monitor = PerformanceMonitor()
        
        # Build pipelines
        self.batch_pipeline = self.orchestrator.build_pipeline()
        self.streaming_pipeline = self.orchestrator.build_streaming_pipeline()
        
        # Current pipeline
        self.current_pipeline = self.batch_pipeline
        
        # Langfuse monitor
        self.langfuse_monitor = None
        
    def process(self, 
                audio_path: str, 
                image_path: str,
                output_path: str = None,
                streaming: bool = False,
                session_id: str = None) -> Dict[str, Any]:
        
        # Generate session ID
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Initialize Langfuse monitor
        if is_langfuse_enabled():
            self.langfuse_monitor = LangfuseMonitor(session_id=session_id)
            self.langfuse_monitor.start_session(
                metadata={
                    "audio_path": audio_path,
                    "image_path": image_path,
                    "output_path": output_path,
                    "streaming": streaming,
                }
            )
        
        # Initialize state
        initial_state: AvatarState = {
            "audio_path": audio_path,
            "image_path": image_path,
            "source_image": self.load_image(image_path),
            "video_frames": [],
            "current_frame_idx": 0,
            "config": self.config,
            "metadata": {"session_id": session_id},
            "errors": [],
            "timestamps": {},
            "performance_metrics": {},
            "streaming_mode": streaming
        }
        
        # Start monitoring
        self.monitor.start_session()
        
        try:
            # Run pipeline with Langfuse callbacks
            pipeline = self.streaming_pipeline if streaming else self.batch_pipeline
            
            # Get Langfuse callbacks if available
            callbacks = getattr(pipeline, '_langfuse_callbacks', [])
            
            if callbacks:
                result = pipeline.invoke(initial_state, config={"callbacks": callbacks})
            else:
                result = pipeline.invoke(initial_state)
            
            # Check for errors
            if result.get("errors"):
                error_msg = f"Pipeline failed: {result['errors']}"
                logger.error(error_msg)
                
                # Log to Langfuse
                if self.langfuse_monitor:
                    self.langfuse_monitor.log_error(error_msg, context=result)
                
                raise RuntimeError(error_msg)
            
            # Save output if needed
            if output_path and "enhanced_frames" in result:
                self.save_video(result["enhanced_frames"], output_path, result.get("audio_path"))
            
            # Log performance
            self.monitor.log_metrics(result.get("timestamps", {}))
            
            # Log to Langfuse
            if self.langfuse_monitor:
                self.langfuse_monitor.log_metrics(result.get("performance", {}))
                self.langfuse_monitor.end_session(final_metrics=result.get("performance", {}))
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.monitor.log_error(str(e))
            
            # Log to Langfuse
            if self.langfuse_monitor:
                self.langfuse_monitor.log_error(str(e))
                self.langfuse_monitor.end_session()
            
            raise
    
    async def process_streaming(self, 
                               audio_stream, 
                               image_path: str,
                               callback=None) -> None:
        
        # Initialize state
        initial_state: AvatarState = {
            "image_path": image_path,
            "source_image": self.load_image(image_path),
            "audio_chunk": [],
            "video_frames": [],
            "current_frame_idx": 0,
            "config": self.config,
            "metadata": {},
            "errors": [],
            "timestamps": {},
            "performance_metrics": {},
            "streaming_mode": True,
            "streaming_complete": False
        }
        
        self.monitor.start_session()
        
        try:
            # Run streaming pipeline
            async for audio_chunk in audio_stream:
                initial_state["audio_chunk"] = audio_chunk
                
                result = self.streaming_pipeline.invoke(initial_state)
                
                # Send frame via callback
                if callback and "streaming_output" in result:
                    await callback(result["streaming_output"])
                
                # Update state for next iteration
                initial_state = result
                
                # Check if streaming should stop
                if result.get("streaming_complete", False):
                    break
                    
        except Exception as e:
            logger.error(f"Streaming pipeline failed: {e}")
            self.monitor.log_error(str(e))
            raise
    
    def load_image(self, image_path: str) -> Any:
        
        img = Image.open(image_path)
        return np.array(img)
    
    def save_video(self, frames: list, output_path: str, audio_path: str = None):
        
        if not frames:
            logger.warning("No frames to save")
            return
        
        # Save video without audio first
        height, width = frames[0].shape[:2]
        temp_path = "temp_video.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        
        # Add audio if provided
        if audio_path:
            video_clip = VideoFileClip(temp_path)
            audio_clip = AudioFileClip(audio_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            video_clip.close()
            audio_clip.close()
        else:
            import shutil
            shutil.move(temp_path, output_path)
        
        logger.info(f"Video saved to: {output_path}")