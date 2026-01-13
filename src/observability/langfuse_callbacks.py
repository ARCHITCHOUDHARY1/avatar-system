"""
Langfuse Callbacks for LangChain/LangGraph Integration
Provides callback handlers for tracing LangGraph workflows
"""

import logging
from typing import Optional, Dict, Any
from .langfuse_config import get_langfuse_client, is_langfuse_enabled, should_trace

logger = logging.getLogger(__name__)


def get_langfuse_callback(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
):
    """
    Get Langfuse callback handler for LangChain/LangGraph
    
    Args:
        session_id: Optional session identifier
        user_id: Optional user identifier
        metadata: Optional metadata to attach to traces
        tags: Optional tags for categorization
    
    Returns:
        CallbackHandler instance or None if Langfuse is disabled
    """
    if not is_langfuse_enabled() or not should_trace():
        return None
    
    try:
        from langfuse.callback import CallbackHandler
        
        client = get_langfuse_client()
        if client is None:
            return None
        
        # Build callback configuration
        callback_config = {}
        
        if session_id:
            callback_config["session_id"] = session_id
        
        if user_id:
            callback_config["user_id"] = user_id
        
        if metadata:
            callback_config["metadata"] = metadata
        
        if tags:
            callback_config["tags"] = tags
        
        # Create callback handler
        handler = CallbackHandler(
            public_key=client.public_key,
            secret_key=client.secret_key,
            host=client.host,
            **callback_config
        )
        
        logger.debug(f"Created Langfuse callback handler (session: {session_id})")
        return handler
        
    except ImportError:
        logger.error("langfuse-langchain not installed. Install with: pip install langfuse-langchain")
        return None
    except Exception as e:
        logger.error(f"Failed to create Langfuse callback: {e}")
        return None


class LangGraphCallbackHandler:
    """
    Custom callback handler for LangGraph-specific tracing
    Wraps Langfuse callback with additional LangGraph context
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.tags = tags or []
        self.langfuse_handler = get_langfuse_callback(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        )
        self.node_traces = {}
    
    def on_node_start(self, node_name: str, state: Dict[str, Any]):
        """Called when a LangGraph node starts execution"""
        if self.langfuse_handler is None:
            return
        
        try:
            # Create a span for this node
            trace_id = f"{self.session_id}_{node_name}" if self.session_id else node_name
            
            logger.debug(f"Node started: {node_name}")
            
            # Store node metadata
            self.node_traces[node_name] = {
                "start_time": __import__("time").time(),
                "state_keys": list(state.keys()) if state else [],
            }
            
        except Exception as e:
            logger.error(f"Error in on_node_start: {e}")
    
    def on_node_end(self, node_name: str, state: Dict[str, Any], output: Any = None):
        """Called when a LangGraph node completes execution"""
        if self.langfuse_handler is None:
            return
        
        try:
            import time
            
            if node_name in self.node_traces:
                duration = time.time() - self.node_traces[node_name]["start_time"]
                logger.debug(f"Node completed: {node_name} (duration: {duration:.2f}s)")
                
                # Clean up
                del self.node_traces[node_name]
            
        except Exception as e:
            logger.error(f"Error in on_node_end: {e}")
    
    def on_node_error(self, node_name: str, error: Exception):
        """Called when a LangGraph node encounters an error"""
        if self.langfuse_handler is None:
            return
        
        try:
            logger.error(f"Node error: {node_name} - {error}")
            
            # Clean up
            if node_name in self.node_traces:
                del self.node_traces[node_name]
            
        except Exception as e:
            logger.error(f"Error in on_node_error: {e}")
    
    def get_handler(self):
        """Get the underlying Langfuse callback handler"""
        return self.langfuse_handler
    
    def flush(self):
        """Flush any pending traces"""
        if self.langfuse_handler:
            try:
                client = get_langfuse_client()
                if client:
                    client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse traces: {e}")
