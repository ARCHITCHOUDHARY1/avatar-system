from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import numpy as np


class AvatarState(TypedDict):
    # Inputs
    audio_input: str  # Audio file path
    image_input: Any  # Reference image path or array
    output_path: str  # Output video path
    
    # Processing
    transcribed_text: str  # Whisper output
    audio_features: Dict[str, Any]  # WavLM features + energy
    emotion: str  # Detected emotion
    confidence: float  # Emotion confidence
    emotion_details: Dict[str, Any]  # Full emotion data
    
    # Mistral Controller Output (NEW)
    avatar_control: Dict[str, float]  # JSON parameters from Mistral
    mistral_response: str  # Full Mistral response
    
    # Output
    final_video: str  # Generated video path
    enhanced_video: str  # GFPGAN enhanced video
    
    # Metadata
    errors: List[str]  # Error messages
    performance: Dict[str, float]  # Performance metrics


class MistralAvatarOrchestrator:
    """
    LangGraph orchestrator with Mistral controller
    Pipeline: Audio -> Emotion -> Mistral Control -> Video -> Enhancement
    """
    
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = StateGraph(AvatarState)
        
    def build_pipeline(self):
        """Build complete pipeline with Mistral controller"""
        
        # Import nodes
        from .workflow_nodes import (
            AudioProcessingNode,
            EmotionDetectionNode,
            MistralControllerNode,
            VideoGenerationNode,
            QualityEnhancementNode
        )
        
        # Create node instances
        audio_node = AudioProcessingNode()
        emotion_node = EmotionDetectionNode()
        mistral_node = MistralControllerNode()  # NEW: Mistral controller
        video_node = VideoGenerationNode()
        quality_node = QualityEnhancementNode()
        
        # Add nodes to graph
        self.graph.add_node("audio_processing", audio_node.process)
        self.graph.add_node("emotion_detection", emotion_node.process)
        self.graph.add_node("mistral_controller", mistral_node.process)  # NEW
        self.graph.add_node("video_generation", video_node.generate)
        self.graph.add_node("quality_enhancement", quality_node.enhance)
        
        # Define edges (pipeline flow)
        self.graph.add_edge(START, "audio_processing")
        self.graph.add_edge("audio_processing", "emotion_detection")
        self.graph.add_edge("emotion_detection", "mistral_controller")  # NEW
        self.graph.add_edge("mistral_controller", "video_generation")  # Mistral -> Video
        self.graph.add_edge("video_generation", "quality_enhancement")
        self.graph.add_edge("quality_enhancement", END)
        
        # Compile with checkpointing and Langfuse callbacks
        from ..observability import get_langfuse_callback
        
        callbacks = []
        langfuse_callback = get_langfuse_callback(
            metadata={"pipeline_type": "batch", "version": "1.0"},
            tags=["avatar-generation", "batch-pipeline"],
        )
        if langfuse_callback:
            callbacks.append(langfuse_callback)
        
        compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        
        # Store callbacks for use during invocation
        compiled_graph._langfuse_callbacks = callbacks
        
        return compiled_graph
    
    def build_streaming_pipeline(self):
        """Build real-time streaming pipeline"""
        from .workflow_nodes import StreamingNode
        
        streaming_node = StreamingNode()
        
        self.graph.add_node("streaming", streaming_node.process)
        self.graph.add_edge(START, "streaming")
        
        # Add conditional edge for continuous streaming
        def should_continue(state: AvatarState):
            if state.get("stream_stop", False):
                return END
            return "streaming"
        
        self.graph.add_conditional_edges(
            "streaming",
            should_continue,
            {
                "streaming": "streaming",
                END: END
            }
        )
        
        # Compile with Langfuse callbacks
        from ..observability import get_langfuse_callback
        
        callbacks = []
        langfuse_callback = get_langfuse_callback(
            metadata={"pipeline_type": "streaming", "version": "1.0"},
            tags=["avatar-generation", "streaming-pipeline"],
        )
        if langfuse_callback:
            callbacks.append(langfuse_callback)
        
        compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        compiled_graph._langfuse_callbacks = callbacks
        
        return compiled_graph
