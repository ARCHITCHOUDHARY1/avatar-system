"""
Langfuse Configuration Module
Provides centralized configuration and client management for Langfuse observability
"""

import os
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled in environment"""
    enabled = os.getenv("ENABLE_LANGFUSE", "false").lower() == "true"
    
    if enabled:
        # Check if required keys are present
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        
        if not public_key or not secret_key:
            logger.warning(
                "ENABLE_LANGFUSE is true but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing. "
                "Langfuse will be disabled. Get your keys from https://cloud.langfuse.com"
            )
            return False
    
    return enabled


@lru_cache(maxsize=1)
def get_langfuse_client():
    """
    Get or create Langfuse client singleton
    Returns None if Langfuse is disabled or not configured
    """
    if not is_langfuse_enabled():
        logger.info("Langfuse is disabled")
        return None
    
    try:
        from langfuse import Langfuse
        
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        release = os.getenv("LANGFUSE_RELEASE", "v1.0.0")
        debug = os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"
        
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release=release,
            debug=debug,
            flush_interval=float(os.getenv("LANGFUSE_FLUSH_INTERVAL", "1")),
        )
        
        logger.info(f"Langfuse client initialized (host: {host}, release: {release})")
        return client
        
    except ImportError:
        logger.error(
            "Langfuse package not installed. Install with: pip install langfuse langfuse-langchain"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None


def get_sample_rate() -> float:
    """Get the configured sample rate for tracing"""
    try:
        rate = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
        return max(0.0, min(1.0, rate))  # Clamp between 0 and 1
    except ValueError:
        logger.warning("Invalid LANGFUSE_SAMPLE_RATE, using 1.0")
        return 1.0


def should_trace() -> bool:
    """Determine if current request should be traced based on sample rate"""
    if not is_langfuse_enabled():
        return False
    
    import random
    return random.random() < get_sample_rate()


def get_environment() -> str:
    """Get current environment"""
    env = os.getenv("ENVIRONMENT", "development")
    enabled_envs = os.getenv("LANGFUSE_ENABLED_ENVIRONMENTS", "development,staging,production")
    
    if env not in enabled_envs.split(","):
        logger.info(f"Langfuse disabled for environment: {env}")
        return None
    
    return env


class LangfuseConfig:
    """Langfuse configuration container"""
    
    def __init__(self):
        self.enabled = is_langfuse_enabled()
        self.client = get_langfuse_client() if self.enabled else None
        self.sample_rate = get_sample_rate()
        self.environment = get_environment()
        self.release = os.getenv("LANGFUSE_RELEASE", "v1.0.0")
    
    def __repr__(self):
        return (
            f"LangfuseConfig(enabled={self.enabled}, "
            f"sample_rate={self.sample_rate}, "
            f"environment={self.environment}, "
            f"release={self.release})"
        )


# Global config instance
_config: Optional[LangfuseConfig] = None


def get_config() -> LangfuseConfig:
    """Get global Langfuse configuration"""
    global _config
    if _config is None:
        _config = LangfuseConfig()
    return _config
