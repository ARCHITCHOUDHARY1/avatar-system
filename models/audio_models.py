from transformers import pipeline, AutoModel, AutoProcessor, AutoTokenizer
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FreeAudioModels:
    """Free HuggingFace models for audio processing - no API keys needed"""
    
    def __init__(self, cache_dir="./models/cache"):
        self.models = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"FreeAudioModels initialized on {self.device}")
    
    def load_mistral_audio(self):
        """Load Mistral for text understanding from transcribed audio"""
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info("Loading Mistral-7B (quantized)...")
            
            # Use smaller version for Colab
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,  # 8-bit quantization for Colab
                cache_dir=str(self.cache_dir)
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                cache_dir=str(self.cache_dir)
            )
            
            self.models['mistral'] = (model, tokenizer)
            logger.info("Mistral loaded successfully")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Mistral: {e}")
            raise
    
    def load_whisper(self, model_size="tiny"):
        """
        Load Whisper for speech-to-text
        Sizes: tiny, base, small, medium, large (bigger = better but slower)
        """
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
            logger.info(f"Whisper {model_size} loaded successfully")
            
            return processor, model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    def load_wav2vec2(self):
        """Load Wav2Vec2 for audio feature extraction"""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            
            logger.info("Loading Wav2Vec2...")
            
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
            
            self.models['wav2vec2'] = (processor, model)
            logger.info("Wav2Vec2 loaded successfully")
            
            return processor, model
            
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2: {e}")
            raise
    
    def load_emotion_detector(self):
        """Load emotion detection model"""
        try:
            logger.info("Loading emotion detector...")
            
            # Emotion detection pipeline
            classifier = pipeline(
                "audio-classification",
                model="superb/hubert-base-superb-er",
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"cache_dir": str(self.cache_dir)}
            )
            
            self.models['emotion'] = classifier
            logger.info("Emotion detector loaded successfully")
            
            return classifier
            
        except Exception as e:
            logger.error(f"Failed to load emotion detector: {e}")
            raise
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        try:
            if 'whisper' not in self.models:
                self.load_whisper(model_size="tiny")
            
            processor, model = self.models['whisper']
            
            # Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
            
            # Decode
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def detect_emotion(self, audio_path):
        """Detect emotion from audio"""
        try:
            if 'emotion' not in self.models:
                self.load_emotion_detector()
            
            classifier = self.models['emotion']
            
            result = classifier(audio_path)
            
            # Get top emotion
            top_emotion = max(result, key=lambda x: x['score'])
            
            logger.info(f"Emotion: {top_emotion['label']} ({top_emotion['score']:.2f})")
            return top_emotion
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return {"label": "neutral", "score": 0.5}
    
    def understand_with_mistral(self, text, prompt_template=None):
        """Use Mistral to understand context and enhance avatar generation"""
        try:
            if 'mistral' not in self.models:
                self.load_mistral_audio()
            
            model, tokenizer = self.models['mistral']
            
            # Default prompt for avatar context
            if prompt_template is None:
                prompt_template = """[INST] Analyze this speech and describe the emotional tone, emphasis points, and suggested facial expressions for an avatar:

Speech: {text}

Provide:
1. Overall emotion
2. Key emphasis words
3. Suggested facial expressions (smile, serious, surprised, sad, etc.)
4. Intensity level (1-10)
[/INST]"""
            
            prompt = prompt_template.format(text=text)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response after [/INST]
            if "[/INST]" in response:
                response = response.split("[/INST]")[1].strip()
            
            logger.info(f"Mistral analysis: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Mistral understanding failed: {e}")
            return "Error: Could not analyze text"
    
    def cleanup(self):
        """Clear GPU memory"""
        try:
            del self.models
            self.models = {}
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Models cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
