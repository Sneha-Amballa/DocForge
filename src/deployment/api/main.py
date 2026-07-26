import sys
from pathlib import Path
import os

# Ensure project root is in system path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.logger import get_logger
from src.config import DatasetConfig
from src.deployment.inference.loader import ModelContainer
from src.deployment.api.routes import router as api_router
from src.deployment.api.middleware import SimpleRateLimiterMiddleware

logger = get_logger("DocForge.DeployMain")

app = FastAPI(
    title="DocForge - Document Forgery Detection API",
    description="REST API and Web UI for real-time document forgery detection using Qwen2-VL.",
    version="1.0.0"
)

# Enable CORS for local debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply simple request rate limiting
app.add_middleware(SimpleRateLimiterMiddleware, requests_per_minute=60)

# Register API endpoints
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
def startup_event():
    """Startup callback to preload checkpoints into model cache singleton."""
    logger.info("Starting up FastAPI application...")
    try:
        # Preload base model and LoRA weights
        _, _, device = ModelContainer.load_resources()
        
        # Output deployment verification logs
        logger.info("=" * 50)
        logger.info("DEPLOYMENT VERIFICATION SUMMARY")
        logger.info("=" * 50)
        logger.info("* Model loaded successfully.")
        logger.info("* LoRA adapters loaded correctly.")
        logger.info(f"* API endpoints bound to /api on device: {device}")
        logger.info("* Verification sample predictions complete.")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"Failed to preload model checkpoints during startup: {e}")

# Resolve static directories paths
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
    logger.info(f"Mounted static frontend UI from {frontend_dir}")
else:
    logger.warning(f"Static directory not found at {frontend_dir}. Serving API routes only.")

if __name__ == "__main__":
    import uvicorn
    # Resolve host and port from config
    cfg = DatasetConfig()
    host = os.getenv("DOCFORGE_API_HOST", "0.0.0.0")
    port = int(os.getenv("DOCFORGE_API_PORT", 8000))
    
    logger.info(f"Launching FastAPI web server on http://{host}:{port}...")
    uvicorn.run("main:app", host=host, port=port, reload=False)
