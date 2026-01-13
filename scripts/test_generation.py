"""
Quick test script to verify avatar generation works end-to-end
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_avatar_generation():
    """Test complete avatar generation pipeline"""
    logger.info("=" * 60)
    logger.info("AVATAR GENERATION TEST")
    logger.info("=" * 60)
    
    # Test files
    image_path = "data/inputs/images/test_avatar.jpg"
    audio_path = "data/inputs/audio/test_audio.wav"
    output_path = "data/outputs/videos/test_output.mp4"
    
    # Check files exist
    if not Path(image_path).exists():
        logger.error(f"? Test image not found: {image_path}")
        return False
    
    if not Path(audio_path).exists():
        logger.error(f"? Test audio not found: {audio_path}")
        return False
    
    logger.info(f"? Test files found")
    logger.info(f"   Image: {image_path}")
    logger.info(f"   Audio: {audio_path}")
    
    try:
        # Import and initialize
        logger.info("\nInitializing SadTalker...")
        from models.sadtalker_integration import EnhancedSadTalker
        
        sadtalker = EnhancedSadTalker()
        logger.info("? SadTalker initialized")
        
        # Load models
        logger.info("\nLoading models...")
        success = sadtalker.load_model()
        if not success:
            logger.error("? Failed to load models")
            return False
        logger.info("? Models loaded")
        
        # Generate video
        logger.info("\nGenerating avatar video...")
        logger.info("? This may take several minutes on CPU...")
        
        video_path = sadtalker.generate_video(
            source_image=image_path,
            driven_audio=audio_path,
            output_path=output_path
        )
        
        logger.info(f"? Video generated: {video_path}")
        
        # Verify output
        if Path(video_path).exists():
            size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            logger.info(f"   File size: {size_mb:.2f}MB")
            logger.info("\n" + "=" * 60)
            logger.info("? TEST PASSED - Avatar generation successful!")
            logger.info("=" * 60)
            return True
        else:
            logger.error("? Output video not found")
            return False
            
    except Exception as e:
        logger.error(f"? Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_avatar_generation()
    sys.exit(0 if success else 1)
