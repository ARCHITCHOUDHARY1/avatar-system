"""
Performance Logging and Monitoring

Tracks execution time and resource usage for each pipeline component
"""

import logging
import time
import psutil
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PerformanceLogger:
    """
    Track performance metrics for avatar generation pipeline
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: List[Dict] = []
        self.node_timings: Dict[str, float] = {}
        self.start_time: Optional[float] = None
        self.memory_start: Optional[float] = None
        
        logger.info(f"PerformanceLogger initialized (session: {self.session_id})")
    
    def start_pipeline(self) -> None:
        """Mark pipeline start"""
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        logger.info("Pipeline started - tracking performance")
    
    def log_node_start(self, node_name: str) -> None:
        """Mark node start time"""
        self.node_timings[f"{node_name}_start"] = time.time()
        logger.debug(f"Node started: {node_name}")
    
    def log_node_end(self, node_name: str, metadata: Optional[Dict] = None) -> float:
        """
        Mark node end and calculate duration
        
        Returns:
            Duration in seconds
        """
        end_time = time.time()
        start_key = f"{node_name}_start"
        
        if start_key not in self.node_timings:
            logger.warning(f"No start time for node: {node_name}")
            return 0.0
        
        duration = end_time - self.node_timings[start_key]
        
        # Record metric
        metric = {
            'node': node_name,
            'duration_seconds': round(duration, 3),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.metrics.append(metric)
        
        logger.info(f"Node completed: {node_name} ({duration:.3f}s)")
        return duration
    
    def end_pipeline(self) -> Dict:
        """
        Mark pipeline end and generate summary
        
        Returns:
            Performance summary dict
        """
        if not self.start_time:
            logger.warning("Pipeline was not started")
            return {}
        
        total_duration = time.time() - self.start_time
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_end - (self.memory_start or 0)
        
        summary = {
            'session_id': self.session_id,
            'total_duration_seconds': round(total_duration, 3),
            'memory_used_mb': round(memory_used, 2),
            'memory_peak_mb': round(memory_end, 2),
            'nodes': {
                metric['node']: metric['duration_seconds']
                for metric in self.metrics
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Pipeline completed in {total_duration:.3f}s, Memory: {memory_used:.2f}MB")
        return summary
    
    def calculate_fps(self, num_frames: int) -> float:
        """
        Calculate effective FPS
        
        Args:
            num_frames: Number of frames generated
            
        Returns:
            Frames per second
        """
        if not self.start_time:
            return 0.0
        
        duration = time.time() - self.start_time
        fps = num_frames / duration if duration > 0 else 0
        
        logger.info(f"Effective FPS: {fps:.2f} ({num_frames} frames in {duration:.2f}s)")
        return fps
    
    def export_metrics(self, output_path: Optional[str] = None) -> str:
        """
        Export metrics to JSON file
        
        Args:
            output_path: Where to save metrics (optional)
            
        Returns:
            Path to saved file
        """
        if not output_path:
            output_dir = Path("data/outputs/logs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"performance_{self.session_id}.json")
        
        summary = self.end_pipeline() if self.start_time else {}
        
        data = {
            'summary': summary,
            'detailed_metrics': self.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to: {output_path}")
        return output_path
    
    def print_summary(self) -> None:
        """Print performance summary to console"""
        summary = self.end_pipeline() if self.start_time else {}
        
        if not summary:
            print("No performance data available")
            return
        
        print("\n" + "=" * 60)
        print(f"Performance Summary - Session: {self.session_id}")
        print("=" * 60)
        print(f"Total Duration: {summary['total_duration_seconds']}s")
        print(f"Memory Used: {summary['memory_used_mb']} MB")
        print(f"Peak Memory: {summary['memory_peak_mb']} MB")
        print("\nNode Timings:")
        for node, duration in summary['nodes'].items():
            print(f"  {node}: {duration}s")
        print("=" * 60 + "\n")


# Context manager for easy timing
class TimedOperation:
    """
    Context manager for timing operations
    
    Usage:
        with TimedOperation("video_generation") as timer:
            generate_video()
        print(f"Took {timer.duration}s")
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - (self.start_time or 0)
        logger.info(f"Completed: {self.operation_name} ({self.duration:.3f}s)")
        return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    perf = PerformanceLogger(session_id="test_123")
    
    # Simulate pipeline
    perf.start_pipeline()
    
    # Simulate nodes
    perf.log_node_start("audio_processing")
    time.sleep(0.1)
    perf.log_node_end("audio_processing", metadata={'format': 'wav'})
    
    perf.log_node_start("video_generation")
    time.sleep(0.2)
    perf.log_node_end("video_generation", metadata={'frames': 100})
    
    # Calculate FPS
    fps = perf.calculate_fps(100)
    
    # Print summary
    perf.print_summary()
    
    # Export
    perf.export_metrics()
    
    # Context manager example
    with TimedOperation("test_operation") as timer:
        time.sleep(0.05)
    print(f"Operation took: {timer.duration:.3f}s")
