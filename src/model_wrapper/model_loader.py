import torch
from pathlib import Path
from typing import Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ModelLoader:
    
    def __init__(self, cache_dir: str = "./models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model_name: str = None,
        device: str = "cuda"
    ) -> Dict:
        
        checkpoint_path = Path(checkpoint_path)
        model_name = model_name or checkpoint_path.stem
        
        # Check cache
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )
            
            # Cache model
            self.loaded_models[model_name] = checkpoint
            
            logger.info(f"Loaded checkpoint: {model_name}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def save_checkpoint(
        self,
        model_state: Dict,
        output_path: str
    ) -> None:
        torch.save(model_state, output_path)
        logger.info(f"Saved checkpoint to: {output_path}")
    
    def clear_cache(self):
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared model cache")
    
    def get_model_info(self, checkpoint_path: str) -> Dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        info = {
            "keys": list(checkpoint.keys()),
            "num_parameters": sum(
                p.numel() for p in checkpoint.values()
                if isinstance(p, torch.Tensor)
            )
        }
        
        return info
