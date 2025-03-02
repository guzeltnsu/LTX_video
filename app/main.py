from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from datetime import datetime
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from app.utils.telemetry import TelemetryManager
import torch

from app.config import Config
from app.models.params import VideoGenerationParams
from app.services.video_generator import VideoGenerator

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Önce config'in geçerli olduğunu kontrol et
try:
    Config.validate()
    logger.info("Configuration validated successfully")
except AssertionError as e:
    logger.error(f"Configuration error: {str(e)}")
    exit(1)

telemetry = TelemetryManager()
# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlatıldığında çalışır
    logger.info("Application startup: Loading model...")
    # Model yükleme işlemini arka planda başlat
    model_task = asyncio.create_task(load_model_async())
    
    # Periyodik metrik loglama işlemini başlat
    metrics_task = asyncio.create_task(log_metrics_periodically())
    
    yield  # Bu noktada FastAPI uygulaması çalışır
    
    # Uygulama kapatıldığında çalışır
    logger.info("Application shutdown: Cleaning up...")
    # Çalışan görevleri iptal et
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Kaynakları temizle
    if generator.model_ready:
        await generator.cleanup_resources()

# Asenkron model yükleme işlevi 
async def load_model_async():
    try:
        generator.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")

# Periyodik metrik loglama işlevi
async def log_metrics_periodically():
    """Belirli aralıklarla sistem metriklerini logla"""
    while True:
        try:
            telemetry.log_system_metrics()
            await asyncio.sleep(60)  # 1 dakikada bir
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            await asyncio.sleep(60)  # Hata durumunda da bekle


# FastAPI uygulamasını lifespan ile başlat
app = FastAPI(title="LTX Video Generation API", lifespan=lifespan)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Video generator instance
generator = VideoGenerator()


@app.post("/generate")
async def generate_video(params: VideoGenerationParams):
    """Video oluşturma endpoint'i"""
    if not generator.model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # RunPod formatına uygun job ID
        job_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        response = await generator.queue_manager.publish(
            job_id=job_id,
            params=params.model_dump()
        )
        
        return {
            "id": response["id"],
            "status": "IN_QUEUE"
        }
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    """Video dosyası indirme endpoint'i"""
    video_path = Config.OUTPUT_DIR / video_name
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path, 
        media_type="video/mp4", 
        filename=video_name
    )

@app.get("/health")
async def health_check():
    """Servis sağlık kontrolü endpoint'i"""
    return {
        "status": "healthy" if generator.model_ready else "degraded",
        "model_status": "ready" if generator.model_ready else "not_ready",
        "device": str(generator.device),
        "queue_status": {
            "queue_length": await generator.queue_manager.get_queue_length()
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Sistem metrikleri endpoint'i"""
    import psutil
    
    metrics = {
        "memory": psutil.virtual_memory()._asdict(),
        "cpu": psutil.cpu_percent(interval=1),
        "queue_length": await generator.queue_manager.get_queue_length()
    }
    
    if generator.device.type == "cuda":
        metrics["gpu"] = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_memory": torch.cuda.max_memory_allocated()
        }
    
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info"
    )