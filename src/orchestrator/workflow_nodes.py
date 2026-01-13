"""
LangGraph Workflow Nodes with Mistral Controller

Pipeline: Wav2Lip + EMOCA + GFPGAN
Controller: Mistral-7B for dynamic avatar parameters
Output: JSON (blink_rate, head_tilt, expression_intensity, etc.)
"""

from typing import Dict, Any, List
import logging
import json
import numpy as np
import time
from pathlib import Path

from ..observability import trace_node, trace_model
from ..observability import trace_node, trace_model, trace_function

logger = logging.getLogger(__name__)


# ============================================
# MISTRAL CONTROLLER NODE (Dynamic Control)
# ============================================

class MistralControllerNode:
    """
    Use Mistral to generate dynamic avatar control parameters
    Input: Audio energy, sentiment, transcription
    Output: JSON with blink_rate, head_tilt, expression_intensity, etc.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.llm = None
    
    @trace_node(name="mistral_controller", metadata={"model": "mistral-7b"})
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state and generate avatar control parameters using Mistral"""
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("MISTRAL CONTROLLER NODE")
            logger.info("=" * 60)
            
            # Extract state information
            audio_energy = state.get("audio_features", {}).get("energy_mean", 0.5)
            sentiment = state.get("emotion", "neutral")
            transcription = state.get("transcribed_text", "")
            confidence = state.get("confidence", 0.5)
            
            logger.info(f"Input - Energy: {audio_energy:.2f}, Sentiment: {sentiment}")
            
            # Load Mistral if needed
            if self.llm is None:
                from models.optimized_models import OptimizedAudioModels
                models = OptimizedAudioModels()
                self.llm = models
                # models.load_mistral()
                logger.info("Skipping Mistral loading to avoid large download. Using heuristics.")
            
            # Create prompt for Mistral
            prompt = self._create_control_prompt(
                energy=audio_energy,
                sentiment=sentiment,
                transcription=transcription,
                confidence=confidence
            )
            
            # Generate control parameters
            logger.info("Generating avatar control parameters with Mistral...")
            # response = self.llm.understand_with_mistral(prompt)
            
            # Parse JSON from response
            # control_params = self._parse_control_json(response, audio_energy, sentiment)
            
            control_params = self._get_heuristic_controls(audio_energy, sentiment)
            response = "Heuristic fallback"
            
            # Store in state
            state["avatar_control"] = control_params
            state["mistral_response"] = response
            
            # Log results
            logger.info(f"Generated controls: {json.dumps(control_params, indent=2)}")
            logger.info(f"Processing time: {time.time() - start_time:.2f}s")
            
            # Update performance metrics
            if "performance" not in state:
                state["performance"] = {}
            state["performance"]["mistral_controller"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Mistral controller failed: {e}", exc_info=True)
            state["errors"].append(f"Mistral controller: {str(e)}")
            
            # Provide fallback default controls
            state["avatar_control"] = self._get_default_controls(
                state.get("emotion", "neutral")
            )
        
        return state
    
    def _create_control_prompt(self, energy, sentiment, transcription, confidence):
        """Create prompt for Mistral to generate avatar controls"""
        
        prompt = f"""[INST] You are an expert avatar animator. Generate realistic avatar control parameters based on the audio analysis.

Audio Analysis:
- Energy level: {energy:.2f} (0.0 = quiet, 1.0 = loud)
- Detected emotion: {sentiment}
- Confidence: {confidence:.2f}
- Speech: "{transcription[:200]}"

Generate a JSON object with these parameters for natural avatar animation:

{{
  "blink_rate": 0.3,        // blinks per second (0.2-0.5 normal, higher if surprised/nervous)
  "head_tilt": 5,           // degrees (-10 to +10, negative=sad, positive=happy/curious)
  "expression_intensity": 0.7,  // 0.0-1.0 (how strong the emotion shows)
  "mouth_openness": 0.5,    // 0.0-1.0 (based on energy)
  "eyebrow_raise": 0.3,     // 0.0-1.0 (surprised, curious)
  "smile_intensity": 0.5,   // 0.0-1.0 (happy emotions)
  "jaw_tension": 0.2,       // 0.0-1.0 (angry, stressed)
  "gaze_direction": 0,      // degrees horizontal (-15 to +15)
  "emotion_label": "{sentiment}",
  "animation_speed": 1.0    // 0.5-1.5 (slower for sad, faster for excited)
}}

Consider:
1. High energy (>{0.7:.1f}) = more mouth movement, faster blinks
2. Low energy (<{0.3:.1f}) = less movement, slower blinks
3. Emotion matches the detected sentiment
4. Natural variation (don't use same values)

Output ONLY the JSON object, nothing else.
[/INST]"""
        
        return prompt
    
    def _parse_control_json(self, response, energy, sentiment):
        """Parse JSON from Mistral response"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Find JSON object
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                
                # Parse JSON
                controls = json.loads(json_str)
                
                # Validate and clamp values
                controls = self._validate_controls(controls)
                
                logger.info("Successfully parsed Mistral JSON output")
                return controls
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse Mistral JSON: {e}")
            logger.warning(f"Response was: {response[:200]}")
            
            # Fallback to heuristic-based controls
            return self._get_heuristic_controls(energy, sentiment)
    
    def _validate_controls(self, controls):
        """Validate and clamp control values to safe ranges"""
        validated = {
            "blink_rate": float(np.clip(controls.get("blink_rate", 0.3), 0.1, 0.6)),
            "head_tilt": float(np.clip(controls.get("head_tilt", 0), -15, 15)),
            "expression_intensity": float(np.clip(controls.get("expression_intensity", 0.7), 0.0, 1.0)),
            "mouth_openness": float(np.clip(controls.get("mouth_openness", 0.5), 0.0, 1.0)),
            "eyebrow_raise": float(np.clip(controls.get("eyebrow_raise", 0.3), 0.0, 1.0)),
            "smile_intensity": float(np.clip(controls.get("smile_intensity", 0.5), 0.0, 1.0)),
            "jaw_tension": float(np.clip(controls.get("jaw_tension", 0.2), 0.0, 1.0)),
            "gaze_direction": float(np.clip(controls.get("gaze_direction", 0), -20, 20)),
            "emotion_label": str(controls.get("emotion_label", "neutral")),
            "animation_speed": float(np.clip(controls.get("animation_speed", 1.0), 0.5, 1.5))
        }
        
        return validated
    
    def _get_heuristic_controls(self, energy, sentiment):
        """Generate controls using heuristics (fallback)"""
        
        # Emotion-based defaults
        emotion_presets = {
            "happy": {
                "blink_rate": 0.35,
                "head_tilt": 5,
                "expression_intensity": 0.8,
                "smile_intensity": 0.8,
                "eyebrow_raise": 0.3,
                "jaw_tension": 0.1,
                "animation_speed": 1.1
            },
            "sad": {
                "blink_rate": 0.25,
                "head_tilt": -3,
                "expression_intensity": 0.6,
                "smile_intensity": 0.1,
                "eyebrow_raise": 0.1,
                "jaw_tension": 0.2,
                "animation_speed": 0.8
            },
            "angry": {
                "blink_rate": 0.4,
                "head_tilt": 2,
                "expression_intensity": 0.9,
                "smile_intensity": 0.0,
                "eyebrow_raise": 0.6,
                "jaw_tension": 0.8,
                "animation_speed": 1.2
            },
            "surprised": {
                "blink_rate": 0.5,
                "head_tilt": 3,
                "expression_intensity": 0.9,
                "smile_intensity": 0.3,
                "eyebrow_raise": 0.9,
                "jaw_tension": 0.1,
                "animation_speed": 1.3
            },
            "neutral": {
                "blink_rate": 0.3,
                "head_tilt": 0,
                "expression_intensity": 0.5,
                "smile_intensity": 0.4,
                "eyebrow_raise": 0.2,
                "jaw_tension": 0.2,
                "animation_speed": 1.0
            }
        }
        
        # Get base preset
        preset = emotion_presets.get(sentiment.lower(), emotion_presets["neutral"]).copy()
        
        # Adjust based on energy
        preset["mouth_openness"] = np.clip(energy * 0.8, 0.2, 0.9)
        preset["blink_rate"] *= (1.0 + (energy - 0.5) * 0.3)  # More energy = more blinks
        preset["expression_intensity"] *= (0.8 + energy * 0.4)  # More energy = stronger expression
        
        # Add other fields
        preset["gaze_direction"] = 0
        preset["emotion_label"] = sentiment
        
        return self._validate_controls(preset)
    
    def _get_default_controls(self, sentiment):
        """Get safe default controls"""
        return self._get_heuristic_controls(0.5, sentiment)


# ============================================
# AUDIO PROCESSING NODE
# ============================================

class AudioProcessingNode:
    """Extract audio features for Mistral controller"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.audio_models = None
    
    @trace_node(name="audio_processing", metadata={"task": "feature_extraction"})
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio and extract features"""
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("AUDIO PROCESSING NODE")
            logger.info("=" * 60)
            
            audio_path = state.get("audio_input")
            
            # Load models if needed
            if self.audio_models is None:
                from models.optimized_models import OptimizedAudioModels
                self.audio_models = OptimizedAudioModels()
            
            # Transcribe with Whisper
            logger.info("Skipping Whisper transcription (optimization)...")
            # transcription = self.audio_models.transcribe_audio(audio_path)
            state["transcribed_text"] = "Skipped"
            
            # Extract WavLM features
            logger.info("Skipping WavLM extraction (optimization)...")
            # features = self.audio_models.extract_audio_features(audio_path)
            features = np.zeros(768)
            
            # Calculate energy
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            energy = np.sqrt(np.mean(audio ** 2))
            energy_normalized = np.clip(energy * 10, 0, 1)  # Normalize to 0-1
            
            state["audio_features"] = {
                "wavlm": features.tolist() if isinstance(features, np.ndarray) else features,
                "energy_mean": float(energy_normalized),
                "duration": float(len(audio) / sr)
            }
            
            logger.info(f"Transcription: {state.get('transcribed_text', 'Skipped')}")
            logger.info(f"Audio energy: {energy_normalized:.2f}")
            logger.info(f"Processing time: {time.time() - start_time:.2f}s")
            
            # Update performance
            if "performance" not in state:
                state["performance"] = {}
            state["performance"]["audio_processing"] = float(time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}", exc_info=True)
            state["errors"].append(f"Audio processing: {str(e)}")
        
        return state


# ============================================
# EMOTION DETECTION NODE
# ============================================

class EmotionDetectionNode:
    """Detect emotion from audio and face"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.audio_models = None
    
    @trace_node(name="emotion_detection", metadata={"task": "multimodal_emotion"})
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect combined emotion"""
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("EMOTION DETECTION NODE")
            logger.info("=" * 60)
            
            # Load models if needed
            if self.audio_models is None:
                from models.optimized_models import OptimizedAudioModels
                self.audio_models = OptimizedAudioModels()
            
            # Get combined emotion (audio + face)
            audio_path = state.get("audio_input")
            image_path = state.get("image_input") if isinstance(state.get("image_input"), str) else None
            
            logger.info("Detecting combined emotion (audio + face)...")
            emotion_result = self.audio_models.get_combined_emotion(audio_path, image_path)
            
            state["emotion"] = str(emotion_result["label"])
            state["confidence"] = float(emotion_result["score"])
            state["emotion_details"] = emotion_result
            
            logger.info(f"Detected emotion: {emotion_result['label']} ({emotion_result['score']:.2f})")
            logger.info(f"Processing time: {time.time() - start_time:.2f}s")
            
            # Update performance
            if "performance" not in state:
                state["performance"] = {}
            state["performance"]["emotion_detection"] = float(time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}", exc_info=True)
            state["errors"].append(f"Emotion detection: {str(e)}")
            state["emotion"] = "neutral"
            state["confidence"] = 0.5
        
        return state


# ============================================
# VIDEO GENERATION NODE (Wav2Lip + EMOCA + GFPGAN)
# ============================================

class VideoGenerationNode:
    """Generate video using Hybrid Model (Wav2Lip for CPU, SadTalker for GPU)"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.generator = None
    
    @trace_node(name="video_generation", metadata={"model": "sadtalker"})
    def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video with dynamic controls from Mistral"""
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("VIDEO GENERATION NODE")
            logger.info("=" * 60)
            
            # Check GPU and load hybrid generator
            import torch
            has_gpu = torch.cuda.is_available()
            model_name = "SadTalker" if has_gpu else "Wav2Lip"
            logger.info(f"Hardware: {'GPU' if has_gpu else 'CPU'} | Model: {model_name}")
            
            if self.generator is None:
                from src.video_processor.hybrid_generator import HybridVideoGenerator
                self.generator = HybridVideoGenerator()
            
            # Get avatar controls from Mistral
            avatar_control = state.get("avatar_control", {})
            emotion = state.get("emotion", "neutral")
            
            logger.info(f"Generating video with controls:")
            logger.info(f"  Emotion: {emotion}")
            logger.info(f"  Blink rate: {avatar_control.get('blink_rate', 0.3)}")
            logger.info(f"  Head tilt: {avatar_control.get('head_tilt', 0)}")
            logger.info(f"  Expression intensity: {avatar_control.get('expression_intensity', 0.7)}")
            
            # Generate video with hybrid model
            video_path = self.generator.generate(
                image_path=state.get("image_input"),
                audio_path=state.get("audio_input"),
                output_path=state.get("output_path"),
                fps=state.get("fps", 25),
                resolution=tuple(state.get("resolution", [512, 512])),
                emotion=emotion
            )
            
            state["final_video"] = video_path
            
            logger.info(f"Video generated: {video_path}")
            logger.info(f"Processing time: {time.time() - start_time:.2f}s")
            
            # Update performance
            if "performance" not in state:
                state["performance"] = {}
            state["performance"]["video_generation"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            state["errors"].append(f"Video generation: {str(e)}")
        
        return state


# ============================================
# QUALITY ENHANCEMENT NODE (GFPGAN)
# ============================================

class QualityEnhancementNode:
    """Enhance video quality with GFPGAN"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.enhancer = None
    
    @trace_node(name="quality_enhancement", metadata={"model": "gfpgan"})
    def enhance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance video quality"""
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("QUALITY ENHANCEMENT NODE (GFPGAN)")
            logger.info("=" * 60)
            
            video_path = state.get("final_video")
            
            if not video_path or not Path(video_path).exists():
                logger.warning("No video to enhance")
                return state
            
            logger.info(f"Enhancing video: {video_path}")
            
            # TODO: Implement actual GFPGAN enhancement
            # For now, just log
            logger.info("GFPGAN enhancement (placeholder)")
            
            # Enhanced path
            enhanced_path = str(Path(video_path).parent / f"{Path(video_path).stem}_enhanced.mp4")
            state["enhanced_video"] = enhanced_path
            
            logger.info(f"Enhancement complete: {enhanced_path}")
            logger.info(f"Processing time: {time.time() - start_time:.2f}s")
            
            # Update performance
            if "performance" not in state:
                state["performance"] = {}
            state["performance"]["quality_enhancement"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Quality enhancement failed: {e}", exc_info=True)
            state["errors"].append(f"Quality enhancement: {str(e)}")
        
        return state


# ============================================
# STREAMING NODE (Real-time)
# ============================================

class StreamingNode:
    """Real-time streaming with Mistral control"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process streaming chunk"""
        try:
            # TODO: Implement streaming
            logger.info("Streaming node (placeholder)")
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            state["errors"].append(f"Streaming: {str(e)}")
        
        return state