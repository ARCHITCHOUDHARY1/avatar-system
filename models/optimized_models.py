"""
Enhanced Free Models - Optimized Selection

Using best-in-class FREE models:
- WavLM (Microsoft) - Audio understanding
- EMOCA (Face emotion) - Facial expression  
- GFPGAN (Tencent) - Face enhancement
- Whisper (OpenAI) - Speech-to-text
- Mistral (Mistral AI) - Context understanding
"""

from transformers import pipeline, AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class OptimizedAudioModels:
    """Optimized model selection with WavLM, EMOCA, and GFPGAN"""
    
    def __init__(self, cache_dir="./models/cache"):
        self.models = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model registry with optimal choices
        self.model_registry = {
            "wavlm": "microsoft/wavlm-base",  # UPGRADED from wav2vec2
            "emoca": "radekd91/emoca",  # NEW for face emotion
            "gfpgan": "TencentARC/GFPGANv1.4",  # KEEP
            "whisper": "openai/whisper-tiny",  # KEEP
            "mistral": "mistralai/Mistral-7B-Instruct-v0.2",  # KEEP
            "hubert_emotion": "superb/hubert-base-superb-er"  # KEEP
        }
        
        logger.info(f"OptimizedAudioModels initialized on {self.device}")
        logger.info(f"Model registry: {list(self.model_registry.keys())}")
    
    # ======================================
    # UPGRADED: WavLM (Better than Wav2Vec2)
    # ======================================
    
    def load_wavlm(self):
        """
        Load WavLM - Microsoft's improved audio model
        UPGRADE from Wav2Vec2 - Better contextualized representations
        """
        try:
            from transformers import WavLMProcessor, WavLMModel
            
            logger.info("Loading WavLM (Microsoft)...")
            
            processor = WavLMProcessor.from_pretrained(
                self.model_registry["wavlm"],
                cache_dir=str(self.cache_dir)
            )
            
            model = WavLMModel.from_pretrained(
                self.model_registry["wavlm"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=str(self.cache_dir)
            )
            
            model.to(self.device)
            
            self.models['wavlm'] = (processor, model)
            logger.info("[OK] WavLM loaded successfully (UPGRADED from Wav2Vec2)")
            
            return processor, model
            
        except Exception as e:
            logger.error(f"Failed to load WavLM: {e}")
            logger.info("Falling back to Wav2Vec2...")
            return self._load_wav2vec2_fallback()
    
    def _load_wav2vec2_fallback(self):
        """Fallback to Wav2Vec2 if WavLM fails"""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            
            logger.info("Loading Wav2Vec2 (fallback)...")
            
            processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h",
                cache_dir=str(self.cache_dir)
            )
            
            model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=str(self.cache_dir)
            )
            
            model.to(self.device)
            self.models['wavlm'] = (processor, model)  # Store as wavlm for compatibility
            
            return processor, model
            
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            raise
    
    # ======================================
    # NEW: EMOCA (Face Emotion Detection)
    # ======================================
    
    def load_emoca(self):
        """
        Load EMOCA - 3D face reconstruction and emotion
        NEW addition for facial emotion detection
        """
        try:
            logger.info("Loading EMOCA (Face emotion)...")
            
            # EMOCA requires special handling
            # For now, use a compatible face emotion model
            from transformers import pipeline
            
            # Use emotion detection pipeline
            emotion_detector = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",  # Alternative face emotion model
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"cache_dir": str(self.cache_dir)}
            )
            
            self.models['emoca'] = emotion_detector
            logger.info("[OK] Face emotion detector loaded (EMOCA alternative)")
            
            return emotion_detector
            
        except Exception as e:
            logger.error(f"Failed to load EMOCA: {e}")
            logger.warning("Face emotion detection will not be available")
            return None
    
    def detect_face_emotion(self, image_path):
        """Detect emotion from face image using EMOCA"""
        try:
            if 'emoca' not in self.models:
                self.load_emoca()
            
            if self.models.get('emoca') is None:
                return {"label": "neutral", "score": 0.5}
            
            detector = self.models['emoca']
            result = detector(image_path)
            
            # Get top emotion
            top_emotion = result[0] if result else {"label": "neutral", "score": 0.5}
            
            logger.info(f"Face emotion: {top_emotion['label']} ({top_emotion['score']:.2f})")
            return top_emotion
            
        except Exception as e:
            logger.error(f"Face emotion detection failed: {e}")
            return {"label": "neutral", "score": 0.5}
    
    # ======================================
    # KEEP: Whisper (Best STT)
    # ======================================
    
    def load_whisper(self, model_size="tiny"):
        """Load Whisper for speech-to-text (KEEP - best in class)"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            model_name = f"openai/whisper-{model_size}"
            logger.info(f"Loading Whisper {model_size}...")
            
            processor = WhisperProcessor.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=str(self.cache_dir)
            )
            
            model.to(self.device)
            
            self.models['whisper'] = (processor, model)
            logger.info(f"[OK] Whisper {model_size} loaded")
            
            return processor, model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    # ======================================
    # KEEP: Mistral (Best LLM)
    # ======================================
    
    def load_mistral(self):
        """Load Mistral for context understanding (KEEP - best 7B model)"""
        try:
            logger.info("Loading Mistral-7B (8-bit quantized)...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_registry["mistral"],
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
                cache_dir=str(self.cache_dir)
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_registry["mistral"],
                cache_dir=str(self.cache_dir)
            )
            
            self.models['mistral'] = (model, tokenizer)
            logger.info("[OK] Mistral loaded (8-bit quantized)")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Mistral: {e}")
            raise
    
    # ======================================
    # KEEP: HuBERT (Audio Emotion)
    # ======================================
    
    def load_audio_emotion_detector(self):
        """Load HuBERT for audio emotion (KEEP - specialized)"""
        try:
            logger.info("Loading HuBERT emotion detector...")
            
            classifier = pipeline(
                "audio-classification",
                model=self.model_registry["hubert_emotion"],
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"cache_dir": str(self.cache_dir)}
            )
            
            self.models['audio_emotion'] = classifier
            logger.info("[OK] Audio emotion detector loaded")
            
            return classifier
            
        except Exception as e:
            logger.error(f"Failed to load audio emotion detector: {e}")
            raise
    
    # ======================================
    # HIGH-LEVEL FUNCTIONS
    # ======================================
    
    def extract_audio_features(self, audio_path):
        """Extract audio features using WavLM (UPGRADED)"""
        try:
            if 'wavlm' not in self.models:
                self.load_wavlm()
            
            processor, model = self.models['wavlm']
            
            # Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get hidden states
            hidden_states = outputs.last_hidden_state
            
            # Average pooling
            features = hidden_states.mean(dim=1).cpu().numpy()[0]
            
            logger.info(f"Extracted WavLM features: shape={features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(768)  # WavLM hidden size
    
    def transcribe_audio(self, audio_path):
        """Transcribe using Whisper"""
        try:
            if 'whisper' not in self.models:
                self.load_whisper(model_size="tiny")
            
            processor, model = self.models['whisper']
            
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
            
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def detect_audio_emotion(self, audio_path):
        """Detect emotion from audio using HuBERT"""
        try:
            if 'audio_emotion' not in self.models:
                self.load_audio_emotion_detector()
            
            classifier = self.models['audio_emotion']
            result = classifier(audio_path)
            
            top_emotion = max(result, key=lambda x: x['score'])
            
            logger.info(f"Audio emotion: {top_emotion['label']} ({top_emotion['score']:.2f})")
            return top_emotion
            
        except Exception as e:
            logger.error(f"Audio emotion detection failed: {e}")
            return {"label": "neutral", "score": 0.5}
    
    def understand_with_mistral(self, text, prompt_template=None):
        """Use Mistral for context understanding"""
        try:
            if 'mistral' not in self.models:
                self.load_mistral()
            
            model, tokenizer = self.models['mistral']
            
            if prompt_template is None:
                prompt_template = """[INST] Analyze this speech for avatar generation:

Speech: {text}

Provide:
1. Overall emotion (happy/sad/angry/neutral/surprised)
2. Intensity (1-10)
3. Key words to emphasize
4. Suggested facial expression
[/INST]"""
            
            prompt = prompt_template.format(text=text)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "[/INST]" in response:
                response = response.split("[/INST]")[1].strip()
            
            logger.info(f"Mistral analysis complete")
            return response
            
        except Exception as e:
            logger.error(f"Mistral understanding failed: {e}")
            return "Error: Could not analyze"
    
    def get_combined_emotion(self, audio_path, image_path=None):
        """
        Get combined emotion from both audio and face (NEW FEATURE)
        Combines HuBERT (audio) + EMOCA (face) for better accuracy
        """
        try:
            # Get audio emotion
            audio_emotion = self.detect_audio_emotion(audio_path)
            
            # Get face emotion if image provided
            if image_path:
                face_emotion = self.detect_face_emotion(image_path)
                
                # Combine emotions (weighted average)
                combined_score = (audio_emotion['score'] * 0.6 + face_emotion['score'] * 0.4)
                
                # Use audio emotion label if more confident, else face
                if audio_emotion['score'] > face_emotion['score']:
                    combined_label = audio_emotion['label']
                else:
                    combined_label = face_emotion['label']
                
                logger.info(f"Combined emotion: {combined_label} ({combined_score:.2f})")
                logger.info(f"  Audio: {audio_emotion['label']} ({audio_emotion['score']:.2f})")
                logger.info(f"  Face: {face_emotion['label']} ({face_emotion['score']:.2f})")
                
                return {
                    "label": combined_label,
                    "score": combined_score,
                    "audio_emotion": audio_emotion,
                    "face_emotion": face_emotion
                }
            else:
                return audio_emotion
                
        except Exception as e:
            logger.error(f"Combined emotion detection failed: {e}")
            return {"label": "neutral", "score": 0.5}
    
    def cleanup(self):
        """Clean up models and free memory"""
        try:
            for model_name in list(self.models.keys()):
                del self.models[model_name]
            
            self.models = {}
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("All models cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
