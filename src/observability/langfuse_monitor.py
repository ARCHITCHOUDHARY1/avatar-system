"""
Langfuse Monitor Integration
Combines Langfuse tracing with existing performance monitoring
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

from .langfuse_config import get_langfuse_client, is_langfuse_enabled

logger = logging.getLogger(__name__)


class LangfuseMonitor:
    """
    Enhanced monitoring that integrates Langfuse with performance tracking
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.client = get_langfuse_client() if is_langfuse_enabled() else None
        self.current_trace = None
        self.metrics = {}
        self.start_time = None
    
    def start_session(self, metadata: Optional[Dict[str, Any]] = None):
        """Start a new monitoring session"""
        self.start_time = time.time()
        self.metrics = {}
        
        if self.client:
            try:
                self.current_trace = self.client.trace(
                    name=f"pipeline_session_{self.session_id}",
                    metadata=metadata or {},
                    tags=["pipeline", "session"],
                    session_id=self.session_id,
                )
                logger.info(f"Started Langfuse trace session: {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to start Langfuse trace: {e}")
    
    def log_node_start(self, node_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Log the start of a node execution"""
        if self.client and self.current_trace:
            try:
                self.metrics[node_name] = {
                    "start_time": time.time(),
                    "metadata": metadata or {},
                }
                logger.debug(f"Node started: {node_name}")
            except Exception as e:
                logger.error(f"Failed to log node start: {e}")
    
    def log_node_end(
        self,
        node_name: str,
        output: Optional[Any] = None,
        error: Optional[str] = None,
    ):
        """Log the end of a node execution"""
        if self.client and self.current_trace and node_name in self.metrics:
            try:
                duration = time.time() - self.metrics[node_name]["start_time"]
                
                # Create span for this node
                span = self.current_trace.span(
                    name=node_name,
                    metadata={
                        **self.metrics[node_name]["metadata"],
                        "duration_seconds": duration,
                        "status": "error" if error else "success",
                    },
                )
                
                if error:
                    span.update(output={"error": error})
                elif output:
                    span.update(output={"result": str(output)[:500]})
                
                logger.debug(f"Node completed: {node_name} ({duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"Failed to log node end: {e}")
    
    def log_model_inference(
        self,
        model_name: str,
        input_data: Optional[str] = None,
        output_data: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a model inference operation"""
        if self.client and self.current_trace:
            try:
                generation = self.current_trace.generation(
                    name=model_name,
                    model=model_name,
                    input=input_data[:500] if input_data else "",
                    completion=output_data[:500] if output_data else "",
                    metadata=metadata or {},
                )
                logger.debug(f"Logged model inference: {model_name}")
            except Exception as e:
                logger.error(f"Failed to log model inference: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        if self.client and self.current_trace:
            try:
                self.current_trace.update(
                    metadata={
                        "performance_metrics": metrics,
                    }
                )
                logger.debug(f"Logged metrics: {list(metrics.keys())}")
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
    
    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Log an error"""
        if self.client and self.current_trace:
            try:
                self.current_trace.update(
                    metadata={
                        "error": error,
                        "error_context": context or {},
                    },
                    tags=["error"],
                )
                logger.error(f"Logged error: {error}")
            except Exception as e:
                logger.error(f"Failed to log error to Langfuse: {e}")
    
    def end_session(self, final_metrics: Optional[Dict[str, Any]] = None):
        """End the monitoring session"""
        if self.start_time:
            total_duration = time.time() - self.start_time
            
            if self.client and self.current_trace:
                try:
                    self.current_trace.update(
                        output={
                            "total_duration_seconds": total_duration,
                            "final_metrics": final_metrics or {},
                        }
                    )
                    logger.info(f"Ended Langfuse trace session: {self.session_id} ({total_duration:.2f}s)")
                except Exception as e:
                    logger.error(f"Failed to end Langfuse trace: {e}")
            
            # Flush traces
            self.flush()
    
    def flush(self):
        """Flush any pending traces to Langfuse"""
        if self.client:
            try:
                self.client.flush()
                logger.debug("Flushed Langfuse traces")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse traces: {e}")
    
    def create_score(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Create a score for the current trace"""
        if self.client and self.current_trace:
            try:
                self.current_trace.score(
                    name=name,
                    value=value,
                    comment=comment,
                )
                logger.debug(f"Created score: {name}={value}")
            except Exception as e:
                logger.error(f"Failed to create score: {e}")
    
    def log_quality_metrics(self, metrics: Dict[str, float]):
        """Log quality metrics as scores"""
        if self.client and self.current_trace:
            for metric_name, metric_value in metrics.items():
                try:
                    self.create_score(
                        name=metric_name,
                        value=metric_value,
                    )
                except Exception as e:
                    logger.error(f"Failed to log quality metric {metric_name}: {e}")
