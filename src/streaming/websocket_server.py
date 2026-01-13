"""WebSocket Server - Real-time communication"""

import asyncio
import websockets
import json
from typing import Set, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WebSocketServer:
    """WebSocket server for real-time streaming"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_to_client(
        self,
        websocket: websockets.WebSocketServerProtocol,
        message: Dict[str, Any]
    ):
        """Send message to specific client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister(websocket)
        
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all clients"""
        if self.clients:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_json) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle client connection"""
        await self.register(websocket)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.process_message(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def process_message(
        self,
        websocket: websockets.WebSocketServerProtocol,
        data: Dict[str, Any]
    ):
        """Process incoming message"""
        message_type = data.get("type")
        
        if message_type == "ping":
            await self.send_to_client(websocket, {"type": "pong"})
        elif message_type == "start_generation":
            # Handle generation start
            pass
        elif message_type == "stop_generation":
            # Handle generation stop
            pass
    
    async def start(self):
        """Start WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever
