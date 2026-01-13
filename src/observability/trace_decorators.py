"""
Trace Decorators for Langfuse Integration
Provides reusable decorators for tracing different types of operations
"""

import logging
import time
import functools
from typing import Callable, Optional, Dict, Any
from .langfuse_config import get_langfuse_client, is_langfuse_enabled

logger = logging.getLogger(__name__)


def trace_node(name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace LangGraph node execution
    
    Usage:
        @trace_node(name="audio_processing", metadata={"version": "1.0"})
        def process_audio(state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)
            
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)
            
            node_name = name or func.__name__
            start_time = time.time()
            
            try:
                # Create trace
                trace = client.trace(
                    name=f"node_{node_name}",
                    metadata=metadata or {},
                    tags=["langgraph", "node"],
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration = time.time() - start_time
                trace.update(
                    output={"status": "success", "duration_seconds": duration},
                )
                
                logger.debug(f"Traced node: {node_name} ({duration:.2f}s)")
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                trace.update(
                    output={"status": "error", "error": str(e), "duration_seconds": duration},
                )
                logger.error(f"Node error traced: {node_name} - {e}")
                raise
            
        return wrapper
    return decorator


def trace_model(model_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace model inference
    
    Usage:
        @trace_model(model_name="whisper-tiny", metadata={"task": "transcription"})
        def transcribe_audio(audio_path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)
            
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                # Create generation trace
                generation = client.generation(
                    name=f"model_{model_name}",
                    model=model_name,
                    metadata=metadata or {},
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration = time.time() - start_time
                generation.update(
                    completion=str(result)[:500] if result else "",  # Truncate long outputs
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": duration,
                        "status": "success",
                    },
                )
                
                logger.debug(f"Traced model: {model_name} ({duration:.2f}s)")
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                generation.update(
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": duration,
                        "status": "error",
                        "error": str(e),
                    },
                )
                logger.error(f"Model error traced: {model_name} - {e}")
                raise
            
        return wrapper
    return decorator


def trace_api(api_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace API calls
    
    Usage:
        @trace_api(api_name="mistral_api", metadata={"endpoint": "/v1/chat"})
        def call_mistral(prompt):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)
            
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                # Create span
                span = client.span(
                    name=f"api_{api_name}",
                    metadata=metadata or {},
                    tags=["api"],
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration = time.time() - start_time
                span.update(
                    output={"status": "success", "duration_seconds": duration},
                )
                
                logger.debug(f"Traced API: {api_name} ({duration:.2f}s)")
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                span.update(
                    output={"status": "error", "error": str(e), "duration_seconds": duration},
                )
                logger.error(f"API error traced: {api_name} - {e}")
                raise
            
        return wrapper
    return decorator


def trace_generation(
    name: str,
    input_data: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to trace generation operations (video, audio, etc.)
    
    Usage:
        @trace_generation(name="video_generation", metadata={"resolution": "512x512"})
        def generate_video(image_path, audio_path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)
            
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                # Create generation trace
                generation = client.generation(
                    name=name,
                    input=str(input_data)[:500] if input_data else "",
                    metadata=metadata or {},
                    tags=["generation"],
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration = time.time() - start_time
                generation.update(
                    completion=str(result)[:500] if result else "",
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": duration,
                        "status": "success",
                    },
                )
                
                logger.debug(f"Traced generation: {name} ({duration:.2f}s)")
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                generation.update(
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": duration,
                        "status": "error",
                        "error": str(e),
                    },
                )
                logger.error(f"Generation error traced: {name} - {e}")
                raise
            
        return wrapper
    return decorator


def trace_function(name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Generic decorator to trace any function
    
    Usage:
        @trace_function(name="custom_operation", metadata={"type": "preprocessing"})
        def my_function(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)
            
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)
            
            func_name = name or func.__name__
            start_time = time.time()
            
            try:
                # Create span
                span = client.span(
                    name=func_name,
                    metadata=metadata or {},
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration = time.time() - start_time
                span.update(
                    output={"status": "success", "duration_seconds": duration},
                )
                
                logger.debug(f"Traced function: {func_name} ({duration:.2f}s)")
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                span.update(
                    output={"status": "error", "error": str(e), "duration_seconds": duration},
                )
                logger.error(f"Function error traced: {func_name} - {e}")
                raise
            
        return wrapper
    return decorator
