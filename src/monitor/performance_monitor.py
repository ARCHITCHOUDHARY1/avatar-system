"""Performance Monitor - Track and log pipeline performance"""

import time
import psutil
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    session_id: str
    start_time: float
    end_time: float = 0.0
    total_duration: float = 0.0
    
    # Stage timings
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    cpu_percent: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    gpu_memory_mb: List[float] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor pipeline performance and resource usage"""
    
    def __init__(self):
        self.current_session: PerformanceMetrics = None
        self.sessions: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
        logger.info("PerformanceMonitor initialized")
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new monitoring session"""
        try:
            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = PerformanceMetrics(
                session_id=session_id,
                start_time=time.time()
            )
            
            # Record initial resource usage
            self._record_resources()
            
            logger.info(f"Started monitoring session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise
    
    def end_session(self) -> PerformanceMetrics:
        """End current monitoring session"""
        try:
            if self.current_session is None:
                logger.warning("No active session to end")
                return None
            
            self.current_session.end_time = time.time()
            self.current_session.total_duration = (
                self.current_session.end_time - self.current_session.start_time
            )
            
            # Final resource snapshot
            self._record_resources()
            
            # Save session
            self.sessions.append(self.current_session)
            
            logger.info(f"Ended session: {self.current_session.session_id}, "
                       f"duration: {self.current_session.total_duration:.2f}s")
            
            return self.current_session
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return None
    
    def log_metrics(self, timestamps: Dict[str, float]) -> None:
        """Log stage timings"""
        try:
            if self.current_session is None:
                logger.warning("No active session for logging metrics")
                return
            
            self.current_session.stage_timings.update(timestamps)
            
            # Record resources after each stage
            self._record_resources()
            
            logger.debug(f"Logged metrics: {list(timestamps.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_error(self, error_msg: str) -> None:
        """Log an error"""
        try:
            if self.current_session:
                self.current_session.errors.append(error_msg)
            logger.error(f"Logged error: {error_msg}")
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def log_warning(self, warning_msg: str) -> None:
        """Log a warning"""
        try:
            if self.current_session:
                self.current_session.warnings.append(warning_msg)
            logger.warning(f"Logged warning: {warning_msg}")
        except Exception as e:
            logger.error(f"Failed to log warning: {e}")
    
    def _record_resources(self) -> None:
        """Record current resource usage"""
        try:
            if self.current_session is None:
                return
            
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.current_session.cpu_percent.append(cpu_percent)
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.current_session.memory_mb.append(memory_mb)
            
            # GPU memory (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.current_session.gpu_memory_mb.append(gpu_memory)
            except ImportError:
                pass
            
        except Exception as e:
            logger.debug(f"Resource recording failed: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if self.current_session is None:
                return {}
            
            session = self.current_session
            
            summary = {
                "session_id": session.session_id,
                "total_duration": session.total_duration,
                "stage_timings": session.stage_timings,
                "avg_cpu_percent": sum(session.cpu_percent) / len(session.cpu_percent) if session.cpu_percent else 0,
                "max_cpu_percent": max(session.cpu_percent) if session.cpu_percent else 0,
                "avg_memory_mb": sum(session.memory_mb) / len(session.memory_mb) if session.memory_mb else 0,
                "max_memory_mb": max(session.memory_mb) if session.memory_mb else 0,
                "error_count": len(session.errors),
                "warning_count": len(session.warnings)
            }
            
            if session.gpu_memory_mb:
                summary["avg_gpu_memory_mb"] = sum(session.gpu_memory_mb) / len(session.gpu_memory_mb)
                summary["max_gpu_memory_mb"] = max(session.gpu_memory_mb)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {}
    
    def export_metrics(self, output_path: str) -> None:
        """Export metrics to JSON file"""
        try:
            summary = self.get_summary()
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Metrics exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def print_summary(self) -> None:
        """Print performance summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session ID: {summary.get('session_id', 'N/A')}")
        print(f"Total Duration: {summary.get('total_duration', 0):.2f}s")
        print("\nStage Timings:")
        for stage, duration in summary.get('stage_timings', {}).items():
            print(f"  {stage}: {duration:.2f}s")
        print(f"\nCPU Usage: Avg={summary.get('avg_cpu_percent', 0):.1f}%, "
              f"Max={summary.get('max_cpu_percent', 0):.1f}%")
        print(f"Memory Usage: Avg={summary.get('avg_memory_mb', 0):.1f}MB, "
              f"Max={summary.get('max_memory_mb', 0):.1f}MB")
        
        if 'avg_gpu_memory_mb' in summary:
            print(f"GPU Memory: Avg={summary.get('avg_gpu_memory_mb', 0):.1f}MB, "
                  f"Max={summary.get('max_gpu_memory_mb', 0):.1f}MB")
        
        print(f"\nErrors: {summary.get('error_count', 0)}")
        print(f"Warnings: {summary.get('warning_count', 0)}")
        print("="*60 + "\n")
