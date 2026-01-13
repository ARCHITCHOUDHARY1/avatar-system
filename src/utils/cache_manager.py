"""
Model and Result Caching Manager

Provides in-memory caching for:
- Loaded model instances (avoid reloading)
- Generated video results
- Audio features and embeddings
- Emotion detection results
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple in-memory cache for models and results
    
    Future: Can be extended to use Redis for distributed caching
    """
    
    def __init__(self):
        self._model_cache: Dict[str, Any] = {}
        self._result_cache: Dict[str, Any] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        logger.info("CacheManager initialized")
    
    def cache_model(self, model_name: str, model: Any) -> None:
        """
        Cache a loaded model instance
        
        Args:
            model_name: Unique identifier for the model
            model: Model instance to cache
        """
        self._model_cache[model_name] = model
        logger.info(f"Cached model: {model_name}")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Retrieve cached model
        
        Args:
            model_name: Model identifier
            
        Returns:
            Cached model or None
        """
        if model_name in self._model_cache:
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: {model_name}")
            return self._model_cache[model_name]
        
        self._stats['misses'] += 1
        logger.debug(f"Cache miss: {model_name}")
        return None
    
    def cache_result(self, key: str, result: Any, ttl: Optional[int] = None) -> None:
        """
        Cache a generation result
        
        Args:
            key: Cache key (hash of inputs)
            result: Result to cache
            ttl: Time to live (seconds) - not implemented yet
        """
        self._result_cache[key] = result
        logger.debug(f"Cached result: {key[:16]}...")
    
    def get_result(self, key: str) -> Optional[Any]:
        """
        Retrieve cached result
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None
        """
        if key in self._result_cache:
            self._stats['hits'] += 1
            return self._result_cache[key]
        
        self._stats['misses'] += 1
        return None
    
    @staticmethod
    def generate_key(audio_path: str, image_path: str, **params) -> str:
        """
        Generate cache key from inputs
        
        Args:
            audio_path: Path to audio file
            image_path: Path to image file
            **params: Additional parameters (fps, resolution, etc.)
            
        Returns:
            Hash key for caching
        """
        # Create deterministic key from inputs
        key_data = f"{audio_path}:{image_path}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear cache
        
        Args:
            cache_type: 'models', 'results', or 'all'
        """
        if cache_type in ('models', 'all'):
            self._model_cache.clear()
            logger.info("Cleared model cache")
        
        if cache_type in ('results', 'all'):
            self._result_cache.clear()
            logger.info("Cleared result cache")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics
        
        Returns:
            Dict with hits, misses, evictions
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            **self._stats,
            'total_requests': total,
            'hit_rate_percent': round(hit_rate, 2),
            'cached_models': len(self._model_cache),
            'cached_results': len(self._result_cache)
        }


# Global cache instance
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """
    Get global cache instance (singleton)
    
    Returns:
        CacheManager instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cache = get_cache()
    
    # Cache a model
    cache.cache_model("wav2lip", {"model": "dummy"})
    
    # Retrieve model
    model = cache.get_model("wav2lip")
    print(f"Retrieved model: {model is not None}")
    
    # Generate cache key
    key = CacheManager.generate_key(
        "audio.wav", 
        "image.jpg",
        fps=25,
        resolution=512
    )
    print(f"Cache key: {key}")
    
    # Stats
    print(f"Stats: {cache.get_stats()}")
