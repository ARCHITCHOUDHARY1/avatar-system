"""
Test script for optimized audio models
Tests: VAD, TTS, WavLM, and complete pipeline
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vad():
    """Test Silero VAD"""
    print("\n" + "="*60)
    print("TEST 1: Silero VAD")
    print("="*60)
    
    try:
        from models.vad_detector import SileroVAD
        
        vad = SileroVAD(threshold=0.5)
        print("[OK] VAD loaded successfully")
        
        # Test with sample audio
        import numpy as np
        
        # Speech-like signal
        sample_audio = np.random.randn(16000)  # 1 second
        is_speech = vad.is_speech(sample_audio, sample_rate=16000)
        
        print(f"[OK] Speech detection works: {is_speech}")
        return True
        
    except Exception as e:
        print(f"[ERROR] VAD test failed: {e}")
        return False


def test_tts():
    """Test Chatterbox TTS"""
    print("\n" + "="*60)
    print("TEST 2: Chatterbox TTS")
    print("="*60)
    
    try:
        from models.tts_generator import ChatterboxTTS
        
        tts = ChatterboxTTS()
        print("[OK] TTS loaded successfully")
        
        # Test synthesis
        test_text = "Hello, this is a test of the text to speech system."
        audio = tts.synthesize(test_text)
        
        print(f"[OK] TTS synthesis works: generated {len(audio)} samples")
        return True
        
    except Exception as e:
        print(f"[ERROR] TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    """Test unified speech pipeline"""
    print("\n" + "="*60)
    print("TEST 3: Speech Pipeline")
    print("="*60)
    
    try:
        from models.speech_pipeline import SpeechPipeline
        
        pipeline = SpeechPipeline(use_local_stt=False)  # Use Groq
        print("[OK] Pipeline initialized successfully")
        
        # Test TTS
        text = "Testing the complete speech pipeline."
        audio = pipeline.synthesize_speech(text)
        
        print(f"[OK] Pipeline TTS works: {len(audio)} samples")
        
        # Test emotion (with dummy audio)
        import tempfile
        import soundfile as sf
        import numpy as np
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            dummy_audio = np.random.randn(16000)
            sf.write(f.name, dummy_audio, 16000)
            temp_path = f.name
        
        try:
            emotion = pipeline.detect_emotion(temp_path)
            print(f"[OK] Pipeline emotion detection works: {emotion['emotion']}")
        finally:
            Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("OPTIMIZED AUDIO MODELS - TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test each component
    results['VAD'] = test_vad()
    results['TTS'] = test_tts()
    results['Pipeline'] = test_pipeline()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "[OK] PASS" if passed else "[ERROR] FAIL"
        print(f"{name:15} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[WARNING] SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
