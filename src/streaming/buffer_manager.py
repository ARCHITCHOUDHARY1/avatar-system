"""Buffer Manager - Manage streaming buffers"""

from collections import deque
from typing import Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class BufferManager:
    """Manage buffers for streaming"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
        
    async def put(self, item: Any):
        """Add item to buffer"""
        async with self.lock:
            self.buffer.append(item)
            
    async def get(self) -> Optional[Any]:
        """Get item from buffer"""
        async with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None
    
    async def peek(self) -> Optional[Any]:
        """Peek at first item without removing"""
        async with self.lock:
            if self.buffer:
                return self.buffer[0]
            return None
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) >= self.max_size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    async def clear(self):
        """Clear buffer"""
        async with self.lock:
            self.buffer.clear()
