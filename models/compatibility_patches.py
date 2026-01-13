"""
Patch file to fix torchvision compatibility issues in SadTalker
This should be run BEFORE importing any SadTalker modules
"""

import sys
import logging

logger = logging.getLogger(__name__)

def patch_torchvision():
    """
    Comprehensive patch for torchvision compatibility
    Handles multiple versions and import paths
    """
    
    # Patch 1: torchvision.transforms.functional_tensor
    try:
        import torchvision.transforms.functional_tensor
        logger.debug("torchvision.transforms.functional_tensor already available")
    except (ImportError, AttributeError) as e:
        logger.info(f"Patching torchvision.transforms.functional_tensor ({e})")
        
        # Try method 1: Use _functional_tensor (newer torchvision)
        try:
            import torchvision.transforms._functional_tensor as FT
            sys.modules["torchvision.transforms.functional_tensor"] = FT
            logger.info("? Patched using _functional_tensor")
        except (ImportError, AttributeError):
            # Try method 2: Use functional (older torchvision)
            try:
                import torchvision.transforms.functional as F
                sys.modules["torchvision.transforms.functional_tensor"] = F
                logger.info("? Patched using functional")
            except ImportError:
                logger.warning("??  Could not patch torchvision.transforms.functional_tensor")
    
    # Patch 2: numpy compatibility
    import numpy as np
    if not hasattr(np, 'float'):
        np.float = float
        logger.debug("Patched np.float")
    if not hasattr(np, 'int'):
        np.int = int
        logger.debug("Patched np.int")
    if not hasattr(np, 'bool'):
        np.bool = bool
        logger.debug("Patched np.bool")
    if not hasattr(np, 'complex'):
        np.complex = complex
        logger.debug("Patched np.complex")
    
    logger.info("? All compatibility patches applied")

# Auto-apply patches when this module is imported
patch_torchvision()
