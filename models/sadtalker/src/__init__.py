"""
SadTalker __init__.py with compatibility patches
This ensures patches are applied when SadTalker modules are imported
"""

# Apply compatibility patches BEFORE any other imports
import sys
from pathlib import Path

# Patch torchvision
try:
    import torchvision.transforms.functional_tensor
except (ImportError, AttributeError):
    try:
        import torchvision.transforms._functional_tensor as FT
        sys.modules["torchvision.transforms.functional_tensor"] = FT
    except (ImportError, AttributeError):
        try:
            import torchvision.transforms.functional as F
            sys.modules["torchvision.transforms.functional_tensor"] = F
        except ImportError:
            pass

# Patch numpy
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'complex'):
    np.complex = complex
