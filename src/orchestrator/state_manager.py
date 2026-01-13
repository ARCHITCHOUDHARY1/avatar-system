
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import add_messages
import numpy as np

class AvatarState(TypedDict):
    # Inputs
    audio_chunk: Annotated[List[float], add_messages]  # Streaming audio chunks
    audio_path: Optional[str]  # Full audio file path
    source_image: np.ndarray  # Reference face image
    image_path: Optional[str]  # Image file path
    
    # Processing
    audio_features: Dict[str, Any]  # Extracted audio features
    face_parameters: Dict[str, np.ndarray]  # Generated face parameters
    video_frames: Annotated[List[np.ndarray], add_messages]  # Generated frames
    current_frame_idx: int  # Current frame index
    
    # Metadata
    config: Dict[str, Any]  # Configuration
    metadata: Dict[str, Any]  # Pipeline metadata
    errors: List[str]  # Error messages
    
    # Performance
    timestamps: Dict[str, float]  # Timing information
    performance_metrics: Dict[str, float]  # Performance metrics

class PipelineConfig:
    def __init__(self):
        self.audio_sample_rate = 16000
        self.audio_chunk_size = 16000  # 1 second chunks
        self.video_fps = 30
        self.video_resolution = (512, 512)
        self.temporal_smoothing = True
        self.quality_preset = "medium"  # low, medium, high
        self.streaming_mode = False
        self.max_frames = 900  # 30 seconds max