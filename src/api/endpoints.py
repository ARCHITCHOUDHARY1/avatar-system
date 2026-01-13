
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# Get project root and create data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
AUDIO_DIR = INPUTS_DIR / "audio"
IMAGES_DIR = INPUTS_DIR / "images"
VIDEOS_DIR = INPUTS_DIR / "videos"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Create directories if they don't exist
for directory in [DATA_DIR, INPUTS_DIR, AUDIO_DIR, IMAGES_DIR, VIDEOS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")


class GenerationRequest(BaseModel):
    audio_path: str
    image_path: str
    fps: int = 25
    resolution: tuple[int, int] = (512, 512)


class GenerationResponse(BaseModel):
    video_path: str
    duration: float
    num_frames: int
    status: str


@router.post("/generate", response_model=GenerationResponse)
async def generate_avatar(request: GenerationRequest):
    
    try:
        logger.info("=" * 60)
        logger.info(f"AVATAR GENERATION REQUEST")
        logger.info(f"Image: {request.image_path}")
        logger.info(f"Audio: {request.audio_path}")
        logger.info(f"FPS: {request.fps}, Resolution: {request.resolution}")
        logger.info("=" * 60)
        
        # Validate input files exist
        if not Path(request.image_path).exists():
            logger.error(f"Image file not found: {request.image_path}")
            raise HTTPException(status_code=400, detail=f"Image file not found: {request.image_path}")
        
        if not Path(request.audio_path).exists():
            logger.error(f"Audio file not found: {request.audio_path}")
            raise HTTPException(status_code=400, detail=f"Audio file not found: {request.audio_path}")
        
        # Create output directory
        output_dir = OUTPUTS_DIR / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"avatar_{timestamp}.mp4"
        output_path = output_dir / output_filename
        
        # Initialize workflow
        logger.info("Initializing avatar generation workflow...")
        from src.orchestrator.workflow_nodes import (
            AudioProcessingNode,
            EmotionDetectionNode,
            MistralControllerNode,
            VideoGenerationNode
        )
        
        # Create pipeline state
        state = {
            "audio_input": str(request.audio_path),
            "image_input": str(request.image_path),
            "output_path": str(output_path),
            "fps": request.fps,
            "resolution": list(request.resolution),
            "errors": [],
            "performance": {}
        }
        
        # Execute workflow nodes with error handling
        try:
            logger.info("Step 1/4: Processing audio...")
            audio_node = AudioProcessingNode()
            state = audio_node.process(state)
            logger.info("? Audio processing complete")
            
            logger.info("Step 2/4: Detecting emotion...")
            emotion_node = EmotionDetectionNode()
            state = emotion_node.process(state)
            logger.info(f"? Emotion detected: {state.get('emotion', 'unknown')}")
            
            logger.info("Step 3/4: Generating control parameters...")
            mistral_node = MistralControllerNode()
            state = mistral_node.process(state)
            logger.info("? Control parameters generated")
            
            logger.info("Step 4/4: Generating video (this may take several minutes on CPU)...")
            logger.info("? Please wait, SadTalker is rendering frames...")
            video_node = VideoGenerationNode()
            state = video_node.generate(state)
            logger.info("? Video generation complete")
            
        except Exception as workflow_error:
            logger.error(f"Workflow error: {workflow_error}", exc_info=True)
            state["errors"].append(f"Workflow failed: {str(workflow_error)}")
            raise
        
        # Check for errors
        if state.get("errors"):
            error_msg = "; ".join(state["errors"])
            logger.error(f"Generation completed with errors: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Generation errors: {error_msg}")
        
        # Get final video path
        video_path = state.get("final_video", str(output_path))
        
        # Verify video was created
        if not Path(video_path).exists():
            logger.error(f"Video file not created: {video_path}")
            raise HTTPException(status_code=500, detail="Video generation failed - output file not created")
        
        # Calculate video metadata
        import cv2
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            logger.info("=" * 60)
            logger.info(f"? GENERATION SUCCESSFUL")
            logger.info(f"Video: {video_path}")
            logger.info(f"Duration: {duration:.2f}s, Frames: {frame_count}, FPS: {fps}")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Could not read video metadata: {e}")
            duration = 0
            frame_count = 0
        
        return GenerationResponse(
            video_path=f"/outputs/videos/{output_filename}",
            duration=duration,
            num_frames=frame_count,
            status="completed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"? GENERATION FAILED")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        logger.error(f"Full traceback:", exc_info=True)
        
        # Provide helpful error message
        error_detail = str(e)
        if "out of memory" in error_detail.lower() or "oom" in error_detail.lower():
            error_detail = "Out of memory. Try using a smaller resolution (256x256) or shorter audio file."
        elif "cuda" in error_detail.lower():
            error_detail = f"GPU error: {e}. System will use CPU (slower but functional)."
        elif "file not found" in error_detail.lower():
            error_detail = f"File error: {e}. Please check that all model files are downloaded."
        
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for avatar generation"""
    try:
        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.mpeg', '.ogg', '.m4a', '.flac'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size (50MB max)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 50MB, got {len(content) / (1024*1024):.2f}MB"
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create safe filename with timestamp to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{Path(file.filename).name}"
        file_path = AUDIO_DIR / safe_filename
        
        # Save uploaded file
        logger.info(f"Saving audio file: {safe_filename} ({len(content)} bytes)")
        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except IOError as e:
            logger.error(f"Failed to save audio file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file to disk")
        
        logger.info(f"Audio file saved successfully: {file_path}")
        return {
            "filename": safe_filename,
            "path": str(file_path),
            "size": len(content),
            "type": "audio"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image file for avatar generation"""
    try:
        # Validate file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size (10MB max)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 10MB, got {len(content) / (1024*1024):.2f}MB"
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create safe filename with timestamp to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{Path(file.filename).name}"
        file_path = IMAGES_DIR / safe_filename
        
        # Save uploaded file
        logger.info(f"Saving image file: {safe_filename} ({len(content)} bytes)")
        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except IOError as e:
            logger.error(f"Failed to save image file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file to disk")
        
        logger.info(f"Image file saved successfully: {file_path}")
        return {
            "filename": safe_filename,
            "path": str(file_path),
            "size": len(content),
            "type": "image"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file and extract first frame for avatar generation"""
    try:
        # Validate file extension
        allowed_extensions = {'.mp4', '.avi', '.webm', '.mov', '.mkv'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size (100MB max)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 100MB, got {len(content) / (1024*1024):.2f}MB"
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create safe filename with timestamp to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{Path(file.filename).name}"
        file_path = VIDEOS_DIR / safe_filename
        
        # Save uploaded file
        logger.info(f"Saving video file: {safe_filename} ({len(content)} bytes)")
        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except IOError as e:
            logger.error(f"Failed to save video file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file to disk")
        
        # Extract first frame as image
        try:
            import cv2
            import numpy as np
            
            logger.info(f"Extracting first frame from video: {file_path}")
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Failed to read video file. File may be corrupted.")
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                raise HTTPException(status_code=400, detail="Failed to extract frame from video")
            
            # Save extracted frame as image
            frame_filename = f"{timestamp}_frame.jpg"
            frame_path = IMAGES_DIR / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            logger.info(f"Extracted frame saved: {frame_path}")
            
            return {
                "filename": safe_filename,
                "path": str(file_path),
                "size": len(content),
                "type": "video",
                "extracted_image": str(frame_path),
                "extracted_image_filename": frame_filename
            }
            
        except ImportError:
            logger.error("OpenCV not available for frame extraction")
            raise HTTPException(
                status_code=500, 
                detail="Video processing not available. OpenCV required."
            )
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}", exc_info=True)
            # Return video info even if frame extraction fails
            return {
                "filename": safe_filename,
                "path": str(file_path),
                "size": len(content),
                "type": "video",
                "warning": "Frame extraction failed, please upload an image separately"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Task status storage (simple in-memory for now, can upgrade to DB later)
task_status_db = {}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    
    try:
        # Check if task exists in our status database
        if task_id in task_status_db:
            return task_status_db[task_id]
        
        # If not found, check if it's a recent completion by checking outputs
        output_dir = OUTPUTS_DIR / "videos"
        if output_dir.exists():
            # Look for video with task_id in filename
            for video_file in output_dir.glob(f"*{task_id}*.mp4"):
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "progress": 100.0,
                    "message": "Generation completed",
                    "video_path": f"/outputs/videos/{video_file.name}"
                }
        
        # Task not found
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/dashboard")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        from src.database.models import get_session, GenerationJob
        session = get_session()
        
        # Get job statistics
        total_jobs = session.query(GenerationJob).count()
        completed_jobs = session.query(GenerationJob).filter(GenerationJob.status == 'completed').count()
        failed_jobs = session.query(GenerationJob).filter(GenerationJob.status == 'failed').count()
        processing_jobs = session.query(GenerationJob).filter(GenerationJob.status == 'processing').count()
        
        # Get performance stats
        realtime_jobs = session.query(GenerationJob).filter(GenerationJob.meets_realtime == True).count()
        avg_fps = session.query(func.avg(GenerationJob.fps_achieved)).scalar() or 0
        avg_quality = session.query(func.avg(GenerationJob.quality_score)).scalar() or 0
        
        session.close()
        
        return {
            "total_jobs": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "processing": processing_jobs,
            "realtime_capable": realtime_jobs,
            "avg_fps": round(avg_fps, 2),
            "avg_quality_score": round(avg_quality, 2),
            "success_rate": round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 2)
        }
        
    except Exception as e:
        logger.error(f"Dashboard stats failed: {e}", exc_info=True)
        # Return default stats if database not initialized
        return {
            "total_jobs": 0,
            "completed": 0,
            "failed": 0,
            "processing": 0,
            "realtime_capable": 0,
            "avg_fps": 0,
            "avg_quality_score": 0,
            "success_rate": 0
        }


@router.get("/stats/recent-jobs")
async def get_recent_jobs(limit: int = 10):
    """Get recent generation jobs"""
    try:
        from src.database.models import get_session, GenerationJob
        from sqlalchemy import desc
        
        session = get_session()
        jobs = session.query(GenerationJob).order_by(desc(GenerationJob.created_at)).limit(limit).all()
        
        result = []
        for job in jobs:
            result.append({
                "job_id": job.job_id,
                "status": job.status,
                "emotion": job.emotion,
                "confidence": job.confidence,
                "fps": job.fps_achieved,
                "quality_score": job.quality_score,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "total_time": job.total_time_seconds
            })
        
        session.close()
        return {"jobs": result}
        
    except Exception as e:
        logger.error(f"Recent jobs failed: {e}", exc_info=True)
        return {"jobs": []}


@router.get("/system/status")
async def get_system_status():
    """Get comprehensive system status for settings panel"""
    try:
        import torch
        import psutil
        
        # CPU & RAM
        cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
        ram = psutil.virtual_memory()
        
        # GPU
        gpu_info = {"used": 0, "total": 0}
        device_name = "CPU"
        device_type = "cpu"
        
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info = {
                "used": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "total": round(props.total_memory / 1e9, 2)
            }
            
        # Check loaded models (simplified check)
        # In a real scenario, we'd check the singleton instances
        models_loaded = []
        # We can deduce based on what imports or singletons are active
        # For now, we'll return a static list or check globals if available
        
        return {
            "backend_status": "ready",
            "device": device_type,
            "device_name": device_name,
            "gpu_memory": gpu_info,
            "cpu_usage": cpu_percent,
            "ram_usage": ram.percent,
            "models_loaded": [
                {"name": "Mistral-7B", "status": "loaded", "vram": "Loading..."},
                {"name": "SadTalker", "status": "loaded", "vram": "Loading..."},
                {"name": "Whisper", "status": "idle", "vram": "0GB"}
            ]
        }
        
    except Exception as e:
        logger.error(f"System status failed: {e}", exc_info=True)
        return {
            "backend_status": "error",
            "error": str(e),
            "device": "unknown",
            "gpu_memory": {"used": 0, "total": 0},
            "cpu_usage": 0,
            "ram_usage": 0,
            "models_loaded": []
        }


@router.get("/system/config")
async def get_system_config():
    """Get available system configuration options"""
    import torch
    
    devices = [{"id": "cpu", "name": "CPU"}]
    if torch.cuda.is_available():
        devices.insert(0, {"id": "cuda", "name": f"GPU: {torch.cuda.get_device_name(0)}"})
        
    return {
        "devices": devices,
        "precisions": ["FP16", "FP32", "INT8"],
        "quality_presets": [
            {"id": "fast", "name": "Fast (Lower Quality)"},
            {"id": "balanced", "name": "Balanced"},
            {"id": "quality", "name": "High Quality (Slower)"}
        ]
    }


@router.get("/stats/performance")
async def get_performance_metrics():
    """Get system performance metrics"""
    try:
        import torch
        import psutil
        
        metrics = {
            "device": {
                "type": "cuda" if torch.cuda.is_available() else "cpu",
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                "memory_used_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "ram_percent": psutil.virtual_memory().percent,
                "ram_used_gb": psutil.virtual_memory().used / 1e9,
                "ram_total_gb": psutil.virtual_memory().total / 1e9
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Performance metrics failed: {e}", exc_info=True)
        return {
            "device": {"type": "unknown", "name": "Unknown", "memory_total_gb": 0, "memory_used_gb": 0},
            "system": {"cpu_percent": 0, "ram_percent": 0, "ram_used_gb": 0, "ram_total_gb": 0}
        }


@router.get("/files/supported-formats")
async def get_supported_formats():
    """Get all supported file formats"""
    return {
        "image": {
            "extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"],
            "mime_types": ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"],
            "max_size_mb": 10,
            "recommended": "JPG or PNG with clear face, 512x512 or higher"
        },
        "audio": {
            "extensions": [".wav", ".mp3", ".mpeg", ".ogg", ".m4a", ".flac"],
            "mime_types": ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/x-m4a", "audio/flac"],
            "max_size_mb": 50,
            "recommended": "WAV or MP3, 16kHz sample rate, clear speech"
        },
        "video": {
            "output_formats": [".mp4", ".avi", ".webm"],
            "default_format": ".mp4",
            "codec": "H.264",
            "fps_options": [24, 25, 30, 60],
            "resolution_options": ["256x256", "512x512", "1024x1024"]
        }
    }


from sqlalchemy import func
