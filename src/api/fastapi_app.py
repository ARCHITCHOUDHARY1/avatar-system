
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from .endpoints import router
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    
    app = FastAPI(
        title="Avatar System Orchestrator",
        description="LangGraph-based Avatar Generation System",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    web_dir = project_root / "web"
    static_dir = web_dir / "static"
    outputs_dir = project_root / "data" / "outputs"
    
    # Ensure outputs directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Mounted static files from {static_dir}")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
    
    # Mount outputs directory for serving generated videos
    if outputs_dir.exists():
        app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")
        logger.info(f"Mounted outputs directory from {outputs_dir}")
    else:
        logger.warning(f"Outputs directory not found: {outputs_dir}")
    
    # Include API routers
    app.include_router(router, prefix="/api/v1")
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Avatar System Orchestrator API")
        logger.info(f"Web directory: {web_dir}")
        logger.info(f"Static directory: {static_dir}")
        logger.info(f"Outputs directory: {outputs_dir}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Avatar System Orchestrator API")
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}



    
    @app.get("/")
    async def serve_frontend():
        index_path = web_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "Frontend not found", "path": str(index_path)}
    
    return app


# Create app instance at module level for uvicorn
app = create_app()
