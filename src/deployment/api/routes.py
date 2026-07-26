import io
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image
from typing import Dict, Any, List

from src.logger import get_logger
from src.deployment.inference.loader import ModelContainer
from src.deployment.inference.predictor import DocForgePredictor
from src.deployment.api.schemas import (
    PredictResponse, HealthResponse, ModelInfoResponse,
    MetricsResponse, VersionResponse
)
from src.deployment.api.middleware import sanitize_filename, validate_uploaded_file
from src.training.utils import get_gpu_memory_usage

logger = get_logger("DocForge.DeployRoutes")
router = APIRouter()

# Global statistics counters
REQUEST_STATS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_inference_time_ms": 0.0
}

def get_predictor() -> DocForgePredictor:
    """Helper to fetch and initialize cached predictor.

    Returns:
        DocForgePredictor: Predictor wrapper.
    """
    model, processor, device = ModelContainer.load_resources()
    return DocForgePredictor(model, processor, device)

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Upload document and detect tamper modifications"
)
async def predict_tampering(file: UploadFile = File(...)) -> PredictResponse:
    """Upload document image and run Qwen2-VL forgery analysis."""
    REQUEST_STATS["total_requests"] += 1
    
    # 1. Filename sanitization
    clean_filename = sanitize_filename(file.filename or "uploaded_doc.png")
    
    # 2. File size checking
    # Read files to determine length
    try:
        content = await file.read()
        file_size = len(content)
        
        # Validate MIME type and upload boundaries
        validate_uploaded_file(file.content_type or "image/png", file_size)
    except HTTPException as e:
        REQUEST_STATS["failed_requests"] += 1
        raise e
    except Exception as e:
        REQUEST_STATS["failed_requests"] += 1
        logger.error(f"Upload file reading failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file."
        )

    # 3. Predict image
    try:
        # Load image with Pillow
        image = Image.open(io.BytesIO(content))
        
        # Load Predictor
        predictor = get_predictor()
        
        # Run prediction
        result = predictor.predict_image(image)
        
        # Log success and update timing metrics
        REQUEST_STATS["successful_requests"] += 1
        REQUEST_STATS["total_inference_time_ms"] += result["processing_time_ms"]
        
        logger.info(
            f"Successfully processed upload '{clean_filename}' in "
            f"{result['processing_time_ms']} ms (Result: Tampered={result['tampered']})"
        )
        
        return PredictResponse(**result)
        
    except Exception as e:
        REQUEST_STATS["failed_requests"] += 1
        logger.error(f"Inference failure on uploaded image '{clean_filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference engine error: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System check endpoint"
)
async def health_check() -> HealthResponse:
    """Check application resources availability."""
    model, _, device = ModelContainer.load_resources()
    gpu_mem = get_gpu_memory_usage()
    
    status_str = "ok" if model is not None else "error"
    return HealthResponse(
        status=status_str,
        device=str(device),
        gpu_memory_usage=gpu_mem
    )

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Details model weights metadata"
)
async def model_info() -> ModelInfoResponse:
    """Retrieve details on model parameter divisions."""
    model, _, _ = ModelContainer.load_resources()
    
    # Defaults
    trainable = 16384
    frozen = 159482560
    
    # If loaded PEFT model, sum trainable parameter count
    if model is not None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
    from src.config import DatasetConfig
    cfg = DatasetConfig()
    
    return ModelInfoResponse(
        model_name=cfg.processor_name,
        lora_rank=8,
        lora_alpha=16,
        precision=cfg.precision,
        trainable_parameters=trainable,
        frozen_parameters=frozen
    )

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Profiling latency metrics summaries"
)
async def metrics_summary() -> MetricsResponse:
    """Retrieve real-time request counts and average latencies."""
    success = REQUEST_STATS["successful_requests"]
    total_time = REQUEST_STATS["total_inference_time_ms"]
    
    avg_latency = total_time / success if success > 0 else 0.0
    load_time = ModelContainer.get_loading_time_ms()
    
    return MetricsResponse(
        total_requests=REQUEST_STATS["total_requests"],
        successful_requests=success,
        failed_requests=REQUEST_STATS["failed_requests"],
        avg_inference_time_ms=avg_latency,
        model_loading_time_ms=load_time
    )

@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Retrieve application version details"
)
async def get_version() -> VersionResponse:
    """Check release manifests."""
    return VersionResponse()
