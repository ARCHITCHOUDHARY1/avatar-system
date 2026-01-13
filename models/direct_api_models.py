"""
Direct API Models - Using Official APIs (Better than HuggingFace!)

APIs Used:
1. Mistral AI - Official Mistral API (FREE tier)
2. Groq - Ultra-fast Whisper (FREE, 20x faster!)
3. Replicate - Video generation (SadTalker, Wav2Lip)
4. Assembly AI - Speech analysis (Optional)

NO HuggingFace Inference API needed!
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Check which APIs are available
try:
    from mistralai.client import MistralClient
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logger.warning("mistralai not installed. Run: pip install mistralai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed. Run: pip install groq")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("replicate not installed. Run: pip install replicate")


class DirectAPIModels:
    """
    Use official direct APIs instead of HuggingFace Inference API
    Much more reliable and professional!
    """
    
    def __init__(self):
        """Initialize all direct API clients"""
        
        # Mistral AI - Official API
        if MISTRAL_AVAILABLE and os.getenv("MISTRAL_API_KEY"):
            try:
                self.mistral = MistralClient(
                    api_key=os.getenv("MISTRAL_API_KEY")
                )
                logger.info("[OK] Mistral AI initialized (Official API)")
            except Exception as e:
                logger.error(f"Mistral AI init failed: {e}")
                self.mistral = None
        else:
            self.mistral = None
            if not os.getenv("MISTRAL_API_KEY"):
                logger.warning("MISTRAL_API_KEY not set")
        
        # Groq - Fast Whisper
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq = Groq(
                    api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info("[OK] Groq initialized (Ultra-fast Whisper)")
            except Exception as e:
                logger.error(f"Groq init failed: {e}")
                self.groq = None
        else:
            self.groq = None
            if not os.getenv("GROQ_API_KEY"):
                logger.warning("GROQ_API_KEY not set")
        
        # Replicate - Video Generation
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if REPLICATE_AVAILABLE and self.replicate_token:
            os.environ["REPLICATE_API_TOKEN"] = self.replicate_token
            logger.info("[OK] Replicate initialized (Video generation)")
        elif not self.replicate_token:
            logger.warning("REPLICATE_API_TOKEN not set")
    
    # ============================================
    # Mistral AI - Text Generation
    # ============================================
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Mistral AI official API
        
        Args:
            prompt: Input prompt
            max_tokens: Max output tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.mistral:
            logger.error("Mistral AI not available")
            return self._fallback_generation(prompt)
        
        try:
            response = self.mistral.chat(
                model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),  # FREE!
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return self._fallback_generation(prompt)
    
    def generate_avatar_parameters(
        self, 
        emotion: str, 
        audio_features: Dict
    ) -> Dict[str, float]:
        """
        Generate avatar parameters using Mistral
        
        Args:
            emotion: Detected emotion
            audio_features: Audio analysis
            
        Returns:
            Avatar parameters
        """
        prompt = f"""Generate facial animation parameters for {emotion} emotion.
Output ONLY a JSON object with these exact keys:
- blink_rate: float (0-1)
- smile_intensity: float (0-1)
- jaw_open: float (0-1)
- head_tilt: float (-0.2 to 0.2)

Example: {{"blink_rate": 0.3, "smile_intensity": 0.8, "jaw_open": 0.4, "head_tilt": 0.0}}"""
        
        try:
            result = self.generate_text(prompt, max_tokens=100, temperature=0.5)
            
            # Try to extract JSON
            import json
            import re
            
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                params = json.loads(json_match.group())
                return params
            
        except Exception as e:
            logger.warning(f"Parameter generation failed: {e}")
        
        # Fallback
        return self._fallback_parameters(emotion)
    
    # ============================================
    # Groq - Speech to Text (20x faster!)
    # ============================================
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Groq (ultra-fast Whisper)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        if not self.groq:
            logger.error("Groq not available")
            return ""
        
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.groq.audio.transcriptions.create(
                    file=audio_file,
                    model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3"),
                    response_format="text"
                )
            
            logger.info(f"[OK] Transcribed: {len(transcription)} chars")
            return str(transcription)
        
        except Exception as e:
            logger.error(f"Groq transcription error: {e}")
            return ""
    
    # ============================================
    # Replicate - Video Generation
    # ============================================
    
    def generate_video(
        self,
        audio_path: str,
        image_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate video using Replicate (SadTalker)
        
        Args:
            audio_path: Input audio file
            image_path: Input face image
            output_path: Output video path (optional)
            
        Returns:
            Output path or URL
        """
        if not REPLICATE_AVAILABLE or not self.replicate_token:
            logger.error("Replicate not available")
            return ""
        
        try:
            model = os.getenv(
                "REPLICATE_SADTALKER_MODEL", 
                "cjwbw/sadtalker:latest"
            )
            
            logger.info(f"Generating video with SadTalker...")
            
            output = replicate.run(
                model,
                input={
                    "source_image": open(image_path, "rb"),
                    "driven_audio": open(audio_path, "rb"),
                    "enhancer": "gfpgan"  # Auto-enhance
                }
            )
            
            if output_path and isinstance(output, str):
                # Download output
                import requests
                response = requests.get(output)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"[OK] Video saved: {output_path}")
                return output_path
            
            return output
        
        except Exception as e:
            logger.error(f"Replicate generation error: {e}")
            return ""
    
    # ============================================
    # Fallback Methods
    # ============================================
    
    def _fallback_generation(self, prompt: str) -> str:
        """Simple fallback for text generation"""
        logger.warning("Using fallback text generation")
        return "Parameters generated"
    
    def _fallback_parameters(self, emotion: str) -> Dict[str, float]:
        """Simple emotional parameter mapping"""
        emotion_map = {
            "happy": {"blink_rate": 0.4, "smile_intensity": 0.9, "jaw_open": 0.5, "head_tilt": 0.05},
            "sad": {"blink_rate": 0.2, "smile_intensity": 0.1, "jaw_open": 0.2, "head_tilt": -0.1},
            "angry": {"blink_rate": 0.3, "smile_intensity": 0.0, "jaw_open": 0.6, "head_tilt": 0.0},
            "surprised": {"blink_rate": 0.5, "smile_intensity": 0.4, "jaw_open": 0.7, "head_tilt": 0.1},
            "neutral": {"blink_rate": 0.3, "smile_intensity": 0.3, "jaw_open": 0.3, "head_tilt": 0.0}
        }
        
        return emotion_map.get(emotion.lower(), emotion_map["neutral"])
    
    # ============================================
    # Status Check
    # ============================================
    
    def check_apis(self) -> Dict[str, bool]:
        """Check which APIs are available"""
        return {
            "mistral": self.mistral is not None,
            "groq": self.groq is not None,
            "replicate": REPLICATE_AVAILABLE and bool(self.replicate_token)
        }


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize direct APIs
    models = DirectAPIModels()
    
    # Check status
    print("\n[SEARCH] API Status:")
    status = models.check_apis()
    for api, available in status.items():
        icon = "[OK]" if available else "[ERROR]"
        print(f"  {icon} {api}: {'Available' if available else 'Not configured'}")
    
    # Test Mistral
    if status["mistral"]:
        print("\n[AI] Testing Mistral AI...")
        result = models.generate_text("Say hello!")
        print(f"  Result: {result}")
        
        params = models.generate_avatar_parameters("happy", {})
        print(f"  Parameters: {params}")
    
    # Test Groq
    if status["groq"]:
        print("\n[AUDIO] Groq Whisper ready for transcription")
    
    # Test Replicate
    if status["replicate"]:
        print("\n[VIDEO] Replicate ready for video generation")
    
    print("\n[OK] Direct APIs configured!")
    print("\nTo use:")
    print("1. Get API keys from guide")
    print("2. Add to .env file")
    print("3. pip install mistralai groq replicate")
