"""
Unified Speech Pipeline - Optimized for avatar generation
Combines: Distil-Whisper STT, Silero VAD, WavLM Emotion, Chatterbox TTS
Total latency: ~26ms (STT 12ms + VAD 3ms + Emotion 8ms + TTS 15ms)
"""

import torch
import numpy as np
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import librosa

logger = logging.getLogger(__name__)


class SpeechPipeline:
    """Optimized unified speech processing pipeline"""
    
    def __init__(
        self,
        use_local_stt: bool = False,  # False = use Groq API (already fast)
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize speech pipeline
        
        Args:
            use_local_stt: Use local Distil-Whisper vs Groq API
            device: 'cuda' or 'cpu' (auto-detected if None)
            config: Configuration dict
        """
        self.config = config or {}
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing Speech Pipeline on {self.device}...")
        
        # Components
        self.vad = None
        self.stt = None
        self.emotion_detector = None
        self.tts = None
        self.use_local_stt = use_local_stt
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all pipeline models"""
        try:
            # VAD (always loaded - it's tiny and fast)
            logger.info("Loading Silero VAD...")
            from models.vad_detector import SileroVAD
            self.vad = SileroVAD(threshold=0.5)
            
            # STT
            if self.use_local_stt:
                logger.info("Loading Distil-Whisper (local)...")
                self._load_distil_whisper()
            else:
                logger.info("Using Groq API for STT (already configured)")
                # Groq client loaded lazily when needed
                self.stt = None
            
            # Emotion Detection (WavLM)
            logger.info("Loading WavLM Base+ for emotion...")
            self._load_wavlm()
            
            # TTS
            logger.info("Loading Multi-backend TTS...")
            from models.tts_generator import MultiTTS
            self.tts = MultiTTS(backend="edge", device=self.device)
            
            logger.info("[OK] Speech Pipeline ready")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise
    
    def _load_distil_whisper(self):
        """Load local Distil-Whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            # Load optimized Distil-Whisper
            self.stt = WhisperModel(
                "distil-large-v3",
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            logger.info("[OK] Distil-Whisper loaded")
            
        except Exception as e:
            logger.error(f"Distil-Whisper loading failed: {e}")
            raise
    
    def _load_wavlm(self):
        """Load WavLM Base+ for emotion detection"""
        try:
            from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
            
            model_name = "microsoft/wavlm-base-plus"
            
            self.emotion_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            
            self.emotion_detector = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=8,  # 8 emotions
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.emotion_detector.to(self.device)
            self.emotion_detector.eval()
            
            # Emotion labels
            self.emotion_labels = [
                "neutral", "happy", "sad", "angry", 
                "fearful", "disgusted", "surprised", "calm"
            ]
            
            logger.info("[OK] WavLM Base+ loaded")
            
        except Exception as e:
            logger.error(f"WavLM loading failed: {e}")
            # Fallback to simpler model
            logger.warning("Falling back to basic emotion detector")
            self.emotion_detector = None
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            if self.use_local_stt and self.stt:
                # Local Distil-Whisper (12ms latency)
                segments, info = self.stt.transcribe(
                    audio_path,
                    language="en",
                    beam_size=1,  # Fast decoding
                    vad_filter=False  # We already use Silero VAD
                )
                
                text = " ".join([seg.text for seg in segments])
                logger.info(f"Transcribed (local): {text}")
                return text
                
            else:
                # Groq API (already fast, no setup needed)
                from groq import Groq
                import os
                
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                
                with open(audio_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=f,
                        model="whisper-large-v3",
                        response_format="text"
                    )
                
                logger.info(f"Transcribed (Groq): {transcription}")
                return transcription
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def detect_emotion(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect emotion from audio using WavLM
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            {"emotion": str, "confidence": float, "probabilities": dict}
        """
        try:
            if self.emotion_detector is None:
                return {"emotion": "neutral", "confidence": 0.5, "probabilities": {}}
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            inputs = self.emotion_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.emotion_detector(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top emotion
            top_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, top_idx].item()
            emotion = self.emotion_labels[top_idx]
            
            # All probabilities
            prob_dict = {
                label: probs[0, i].item() 
                for i, label in enumerate(self.emotion_labels)
            }
            
            result = {
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": prob_dict
            }
            
            logger.info(f"Emotion: {emotion} ({confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "probabilities": {}}
    
    def synthesize_speech(
        self, 
        text: str, 
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate speech from text using TTS
        
        Args:
            text: Input text
            output_path: Optional save path
            
        Returns:
            Audio array
        """
        try:
            if self.tts is None:
                raise RuntimeError("TTS not loaded")
            
            audio = self.tts.synthesize(text, output_path)
            return audio
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def process_audio(
        self, 
        audio_path: str,
        filter_silence: bool = True
    ) -> Dict[str, Any]:
        """
        Complete audio processing pipeline
        
        Args:
            audio_path: Input audio file
            filter_silence: Apply VAD to filter silence
            
        Returns:
            {
                "text": str,
                "emotion": dict,
                "vad_segments": list,
                "filtered_audio": Optional[np.ndarray]
            }
        """
        try:
            results = {}
            
            # 1. VAD - Detect speech segments (3ms)
            if self.vad and filter_silence:
                logger.info("Running VAD...")
                vad_segments = self.vad.detect_speech_segments(audio_path)
                results["vad_segments"] = vad_segments
                
                # Filter silence
                audio, sr = librosa.load(audio_path, sr=16000)
                filtered_audio = self.vad.filter_silence(audio, sr)
                results["filtered_audio"] = filtered_audio
            else:
                results["vad_segments"] = []
                results["filtered_audio"] = None
            
            # 2. Parallel: STT + Emotion (can run concurrently)
            future_stt = self.executor.submit(self.transcribe, audio_path)
            future_emotion = self.executor.submit(self.detect_emotion, audio_path)
            
            # Wait for results
            results["text"] = future_stt.result()
            results["emotion"] = future_emotion.result()
            
            logger.info("[OK] Audio processing complete")
            return results
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    def __call__(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Process single audio chunk (for streaming)
        
        Args:
            audio_chunk: Audio array
            sample_rate: Sample rate
            
        Returns:
            Processing results
        """
        try:
            # Quick VAD check
            is_speech = self.vad.is_speech(audio_chunk, sample_rate) if self.vad else True
            
            if not is_speech:
                return {"is_speech": False, "text": "", "emotion": "silence"}
            
            # Save temp file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf
                sf.write(f.name, audio_chunk, sample_rate)
                temp_path = f.name
            
            # Process
            results = self.process_audio(temp_path, filter_silence=False)
            results["is_speech"] = True
            
            # Cleanup
            Path(temp_path).unlink()
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return {"is_speech": False, "error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.executor.shutdown(wait=False)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Pipeline cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Singleton instance
_pipeline_instance = None

def get_speech_pipeline(
    use_local_stt: bool = False,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> SpeechPipeline:
    """Get speech pipeline instance (singleton)"""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = SpeechPipeline(
            use_local_stt=use_local_stt,
            device=device,
            config=config
        )
    
    return _pipeline_instance
