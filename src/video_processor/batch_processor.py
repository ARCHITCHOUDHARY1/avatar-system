"""
Batch Processing System

Handles multiple avatar generation requests with queue management
Preparation for concurrent request handling
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import queue
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Request processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    """Single avatar generation request"""
    request_id: str
    audio_path: str
    image_path: str
    output_path: str
    params: Dict = None
    status: RequestStatus = RequestStatus.PENDING
    created_at: str = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class BatchProcessor:
    """
    Queue-based batch processor for avatar generation
    
    Features:
    - Request queue management
    - Progress tracking
    - Status monitoring
    - Error handling
    
    Future: Add parallel processing with multiprocessing/threading
    """
    
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.request_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.requests: Dict[str, GenerationRequest] = {}
        self.processing_count = 0
        self.completed_count = 0
        self.failed_count = 0
        
        logger.info(f"BatchProcessor initialized (max queue: {max_queue_size})")
    
    def add_request(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        **params
    ) -> str:
        """
        Add generation request to queue
        
        Args:
            audio_path: Path to audio file
            image_path: Path to image file
            output_path: Where to save output
            **params: Additional parameters (fps, resolution, etc.)
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        request = GenerationRequest(
            request_id=request_id,
            audio_path=audio_path,
            image_path=image_path,
            output_path=output_path,
            params=params
        )
        
        try:
            self.request_queue.put(request, block=False)
            self.requests[request_id] = request
            logger.info(f"Added request to queue: {request_id}")
            return request_id
        except queue.Full:
            logger.error("Queue is full, cannot add request")
            raise RuntimeError("Request queue is full")
    
    def get_next_request(self) -> Optional[GenerationRequest]:
        """
        Get next request from queue
        
        Returns:
            GenerationRequest or None if queue empty
        """
        try:
            request = self.request_queue.get(block=False)
            request.status = RequestStatus.PROCESSING
            self.processing_count += 1
            logger.info(f"Processing request: {request.request_id}")
            return request
        except queue.Empty:
            return None
    
    def complete_request(
        self,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Mark request as completed
        
        Args:
            request_id: Request identifier
            success: Whether request succeeded
            error: Error message if failed
        """
        if request_id not in self.requests:
            logger.warning(f"Unknown request ID: {request_id}")
            return
        
        request = self.requests[request_id]
        request.completed_at = datetime.now().isoformat()
        
        if success:
            request.status = RequestStatus.COMPLETED
            self.completed_count += 1
            logger.info(f"Request completed: {request_id}")
        else:
            request.status = RequestStatus.FAILED
            request.error = error
            self.failed_count += 1
            logger.error(f"Request failed: {request_id} - {error}")
        
        self.processing_count -= 1
    
    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get request status
        
        Args:
            request_id: Request identifier
            
        Returns:
            Status dict or None
        """
        if request_id not in self.requests:
            return None
        
        request = self.requests[request_id]
        return {
            'request_id': request.request_id,
            'status': request.status.value,
            'created_at': request.created_at,
            'completed_at': request.completed_at,
            'error': request.error
        }
    
    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics
        
        Returns:
            Stats dict with counts
        """
        return {
            'queue_size': self.request_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'processing': self.processing_count,
            'completed': self.completed_count,
            'failed': self.failed_count,
            'total_requests': len(self.requests)
        }
    
    def clear_completed(self) -> int:
        """
        Clear completed requests from memory
        
        Returns:
            Number of requests cleared
        """
        to_remove = [
            req_id for req_id, req in self.requests.items()
            if req.status in (RequestStatus.COMPLETED, RequestStatus.FAILED)
        ]
        
        for req_id in to_remove:
            del self.requests[req_id]
        
        logger.info(f"Cleared {len(to_remove)} completed requests")
        return len(to_remove)


# Global batch processor instance
_global_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """
    Get global batch processor instance (singleton)
    
    Returns:
        BatchProcessor instance
    """
    global _global_processor
    if _global_processor is None:
        _global_processor = BatchProcessor()
    return _global_processor


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = BatchProcessor(max_queue_size=10)
    
    # Add requests
    req1 = processor.add_request(
        "audio1.wav",
        "image1.jpg",
        "output1.mp4",
        fps=25
    )
    
    req2 = processor.add_request(
        "audio2.wav",
        "image2.jpg",
        "output2.mp4",
        fps=30
    )
    
    print(f"Queue stats: {processor.get_queue_stats()}")
    
    # Process requests
    request = processor.get_next_request()
    if request:
        print(f"Processing: {request.request_id}")
        # Simulate processing
        processor.complete_request(request.request_id, success=True)
    
    # Check status
    status = processor.get_status(req1)
    print(f"Request status: {status}")
    
    # Final stats
    print(f"Final stats: {processor.get_queue_stats()}")
