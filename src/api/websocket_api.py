
from fastapi import WebSocket, WebSocketDisconnect
import json
import logging

logger = logging.getLogger(__name__)


async def websocket_endpoint(websocket: WebSocket):
    
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "start_generation":
                # Start avatar generation
                await websocket.send_json({
                    "type": "status",
                    "message": "Generation started"
                })
                
                # TODO: Implement streaming generation
                for progress in range(0, 101, 10):
                    await websocket.send_json({
                        "type": "progress",
                        "progress": progress
                    })
                
                await websocket.send_json({
                    "type": "complete",
                    "video_path": "/output/video.mp4"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
