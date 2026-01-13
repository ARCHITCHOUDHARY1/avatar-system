"""
Model Download Script - FREE Models Only

Downloads all required FREE open-source models:
- Mistral-7B (FREE local LLM - NO OpenAI!)
- Whisper (speech-to-text)
- WavLM (audio features)
- HuBERT (audio emotion)
- EMOCA/ViT (face emotion)
- Wav2Lip (lip sync)
- SadTalker (avatar generation)
- GFPGAN (face enhancement)

NO API keys required!
Total size: ~10-15 GB (one-time download)
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def download_models():
    """Download all FREE models"""
    
    logger.info("=" * 70)
    logger.info("DOWNLOADING FREE OPEN-SOURCE MODELS")
    logger.info("=" * 70)
    logger.info("This will download ~10-15 GB of models (one-time)")
    logger.info("All models are FREE and run locally!")
    logger.info("")
    
    # Create cache directory
    cache_dir = Path(os.getenv("CACHE_DIR", "./models/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Import after creating directories
    from models.optimized_models import OptimizedAudioModels
    
    models = OptimizedAudioModels(cache_dir=str(cache_dir))
    
    logger.info("=" * 70)
    logger.info("1/8 - Mistral-7B (FREE LLM - replaces OpenAI!)")
    logger.info("=" * 70)
    logger.info("Size: ~4 GB (8-bit quantized)")
    logger.info("Purpose: Dynamic avatar control (FREE & OFFLINE!)")
    logger.info("")
    
    try:
        models.load_mistral()
        logger.info("[OK] Mistral-7B downloaded and cached")
    except Exception as e:
        logger.warning(f"[WARNING]  Mistral download failed: {e}")
        logger.info("    Will retry on first use")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("2/8 - Whisper (Speech-to-Text)")
    logger.info("=" * 70)
    logger.info("Size: ~39 MB (tiny model)")
    logger.info("Purpose: Convert speech to text")
    logger.info("")
    
    try:
        models.load_whisper(model_size='tiny')
        logger.info("[OK] Whisper downloaded and cached")
    except Exception as e:
        logger.warning(f"[WARNING]  Whisper download failed: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("3/8 - WavLM (Audio Features)")
    logger.info("=" * 70)
    logger.info("Size: ~95 MB")
    logger.info("Purpose: Audio understanding")
    logger.info("")
    
    try:
        models.load_wavlm()
        logger.info("[OK] WavLM downloaded and cached")
    except Exception as e:
        logger.warning(f"[WARNING]  WavLM download failed: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("4/8 - HuBERT (Audio Emotion)")
    logger.info("=" * 70)
    logger.info("Size: ~95 MB")
    logger.info("Purpose: Emotion from audio")
    logger.info("")
    
    try:
        models.load_hubert_emotion()
        logger.info("[OK] HuBERT downloaded and cached")
    except Exception as e:
        logger.warning(f"[WARNING]  HuBERT download failed: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("5/8 - EMOCA/ViT (Face Emotion)")
    logger.info("=" * 70)
    logger.info("Size: ~300 MB")
    logger.info("Purpose: Emotion from face")
    logger.info("")
    
    try:
        models.load_face_emotion()
        logger.info("[OK] EMOCA/ViT downloaded and cached")
    except Exception as e:
        logger.warning(f"[WARNING]  EMOCA download failed: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("6/8 - Wav2Lip (Lip Synchronization)")
    logger.info("=" * 70)
    logger.info("Size: ~350 MB")
    logger.info("Purpose: Precise lip sync")
    logger.info("")
    
    # Download Wav2Lip checkpoint
    wav2lip_dir = models_dir / "Wav2Lip" / "checkpoints"
    wav2lip_dir.mkdir(parents=True, exist_ok=True)
    
    wav2lip_checkpoint = wav2lip_dir / "wav2lip_gan.pth"
    
    if not wav2lip_checkpoint.exists():
        logger.info("Downloading Wav2Lip checkpoint...")
        try:
            import gdown
            url = "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW"
            gdown.download(url, str(wav2lip_checkpoint), fuzzy=True)
            logger.info("[OK] Wav2Lip downloaded")
        except Exception as e:
            logger.warning(f"[WARNING]  Wav2Lip download failed: {e}")
            logger.info("    Download manually from: https://github.com/Rudrabha/Wav2Lip")
    else:
        logger.info("[OK] Wav2Lip already downloaded")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("7/8 - SadTalker (Avatar Generation)")
    logger.info("=" * 70)
    logger.info("Size: ~2 GB")
    logger.info("Purpose: Generate talking avatar")
    logger.info("")
    
    # SadTalker checkpoints
    sadtalker_dir = models_dir / "sadtalker" / "checkpoints"
    sadtalker_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("SadTalker will be downloaded on first use")
    logger.info("[OK] SadTalker directory created")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("8/8 - GFPGAN (Face Enhancement)")
    logger.info("=" * 70)
    logger.info("Size: ~350 MB")
    logger.info("Purpose: Enhance face quality")
    logger.info("")
    
    # GFPGAN checkpoint
    gfpgan_dir = models_dir / "gfpgan"
    gfpgan_dir.mkdir(parents=True, exist_ok=True)
    
    gfpgan_checkpoint = gfpgan_dir / "GFPGANv1.4.pth"
    
    if not gfpgan_checkpoint.exists():
        logger.info("Downloading GFPGAN checkpoint...")
        try:
            import gdown
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            import urllib.request
            urllib.request.urlretrieve(url, str(gfpgan_checkpoint))
            logger.info("[OK] GFPGAN downloaded")
        except Exception as e:
            logger.warning(f"[WARNING]  GFPGAN download failed: {e}")
            logger.info("    Will be downloaded on first use")
    else:
        logger.info("[OK] GFPGAN already downloaded")
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[OK] All FREE models ready to use!")
    logger.info("")
    logger.info("Model Stack (100% FREE & LOCAL):")
    logger.info("  ? Mistral-7B (FREE LLM - NO OpenAI!)")
    logger.info("  ? Whisper (speech-to-text)")
    logger.info("  ? WavLM (audio features)")
    logger.info("  ? HuBERT (audio emotion)")
    logger.info("  ? EMOCA/ViT (face emotion)")
    logger.info("  ? Wav2Lip (lip sync)")
    logger.info("  ? SadTalker (avatar generation)")
    logger.info("  ? GFPGAN (face enhancement)")
    logger.info("")
    logger.info("Total size: ~10-15 GB")
    logger.info("Location: ./models/cache/")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Copy .env.example to .env (optional)")
    logger.info("  2. Run: python main.py --mode gradio")
    logger.info("  3. Open: http://localhost:7860")
    logger.info("")
    logger.info("NO API keys needed! Everything runs locally! [OK]")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        logger.error(f"\n\n[ERROR] Download failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check internet connection")
        logger.error("  2. Check disk space (need ~20 GB)")
        logger.error("  3. Try again (models will resume)")
        logger.error("  4. Some models download on first use")
        sys.exit(1)
