"""
SadTalker Diagnostic Script
Checks all prerequisites for avatar generation
"""

import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version"""
    logger.info("=" * 60)
    logger.info("PYTHON VERSION CHECK")
    logger.info("=" * 60)
    version = sys.version_info
    logger.info(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("? Python 3.8+ required")
        return False
    logger.info("? Python version OK")
    return True

def check_dependencies():
    """Check required Python packages"""
    logger.info("\n" + "=" * 60)
    logger.info("DEPENDENCY CHECK")
    logger.info("=" * 60)
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'cv2': 'OpenCV (cv2)',
        'librosa': 'Librosa',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            logger.info(f"? {name} installed")
        except ImportError:
            logger.error(f"? {name} NOT installed")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    logger.info("\n" + "=" * 60)
    logger.info("CUDA CHECK")
    logger.info("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"? CUDA available")
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            logger.warning("??  CUDA not available (will use CPU)")
            return True  # Not critical
    except Exception as e:
        logger.error(f"? Error checking CUDA: {e}")
        return False

def check_sadtalker_files():
    """Check SadTalker model files and source code"""
    logger.info("\n" + "=" * 60)
    logger.info("SADTALKER FILES CHECK")
    logger.info("=" * 60)
    
    base_path = Path("models/sadtalker")
    
    # Check base directory
    if not base_path.exists():
        logger.error(f"? SadTalker directory not found: {base_path}")
        return False
    logger.info(f"? SadTalker directory exists: {base_path}")
    
    # Check checkpoints
    checkpoint_path = base_path / "checkpoints"
    if not checkpoint_path.exists():
        logger.error(f"? Checkpoints directory not found: {checkpoint_path}")
        return False
    logger.info(f"? Checkpoints directory exists")
    
    required_checkpoints = [
        "SadTalker_V0.0.2_256.safetensors",
        "mapping_00109-model.pth.tar"
    ]
    
    all_checkpoints_ok = True
    for checkpoint in required_checkpoints:
        checkpoint_file = checkpoint_path / checkpoint
        if checkpoint_file.exists():
            size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
            logger.info(f"? {checkpoint} ({size_mb:.1f}MB)")
        else:
            logger.error(f"? {checkpoint} NOT FOUND")
            all_checkpoints_ok = False
    
    # Check source code
    src_path = base_path / "src"
    if not src_path.exists():
        logger.error(f"? Source directory not found: {src_path}")
        return False
    logger.info(f"? Source directory exists")
    
    required_src = [
        "facerender",
        "test_audio2coeff.py",
        "utils",
        "config"
    ]
    
    all_src_ok = True
    for item in required_src:
        item_path = src_path / item
        if item_path.exists():
            logger.info(f"? {item}")
        else:
            logger.error(f"? {item} NOT FOUND")
            all_src_ok = False
    
    return all_checkpoints_ok and all_src_ok

def check_data_directories():
    """Check data directories"""
    logger.info("\n" + "=" * 60)
    logger.info("DATA DIRECTORIES CHECK")
    logger.info("=" * 60)
    
    required_dirs = [
        "data/inputs/audio",
        "data/inputs/images",
        "data/inputs/videos",
        "data/outputs/videos"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"? {dir_path}")
        else:
            logger.warning(f"??  {dir_path} not found (will be created)")
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"   Created: {dir_path}")
            except Exception as e:
                logger.error(f"   Failed to create: {e}")
                all_ok = False
    
    return all_ok

def test_sadtalker_import():
    """Test importing SadTalker modules"""
    logger.info("\n" + "=" * 60)
    logger.info("SADTALKER IMPORT TEST")
    logger.info("=" * 60)
    
    try:
        sys.path.insert(0, "models/sadtalker")
        
        logger.info("Attempting to import SadTalker modules...")
        
        from src.facerender.animate import AnimateFromCoeff
        logger.info("? AnimateFromCoeff imported")
        
        from src.test_audio2coeff import Audio2Coeff
        logger.info("? Audio2Coeff imported")
        
        from src.utils.preprocess import CropAndExtract
        logger.info("? CropAndExtract imported")
        
        logger.info("? All SadTalker modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"? Import failed: {e}")
        logger.error("   This usually means:")
        logger.error("   1. SadTalker source code is missing")
        logger.error("   2. SadTalker dependencies not installed")
        logger.error("   3. Incompatible SadTalker version")
        return False
    except Exception as e:
        logger.error(f"? Unexpected error: {e}")
        return False

def test_enhanced_sadtalker():
    """Test loading EnhancedSadTalker"""
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED SADTALKER TEST")
    logger.info("=" * 60)
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from models.sadtalker_integration import EnhancedSadTalker
        logger.info("? EnhancedSadTalker imported")
        
        sadtalker = EnhancedSadTalker()
        logger.info("? EnhancedSadTalker instantiated")
        
        logger.info("Attempting to load models...")
        success = sadtalker.load_model()
        
        if success:
            logger.info("? SadTalker models loaded successfully")
            return True
        else:
            logger.error("? Failed to load SadTalker models")
            return False
            
    except Exception as e:
        logger.error(f"? Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all diagnostic checks"""
    logger.info("\n" + "=" * 60)
    logger.info("SADTALKER DIAGNOSTIC SCRIPT")
    logger.info("=" * 60)
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "CUDA": check_cuda(),
        "SadTalker Files": check_sadtalker_files(),
        "Data Directories": check_data_directories(),
        "SadTalker Import": test_sadtalker_import(),
        "Enhanced SadTalker": test_enhanced_sadtalker()
    }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    for check, passed in results.items():
        status = "? PASS" if passed else "? FAIL"
        logger.info(f"{status} - {check}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("? ALL CHECKS PASSED - System ready for avatar generation")
    else:
        logger.error("? SOME CHECKS FAILED - Please fix issues above")
        logger.error("\nCommon solutions:")
        logger.error("1. Install missing dependencies: pip install -r requirements.txt")
        logger.error("2. Download SadTalker models from official repository")
        logger.error("3. Clone SadTalker source code to models/sadtalker/")
    logger.info("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
