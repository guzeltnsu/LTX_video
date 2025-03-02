import os
from pathlib import Path
from typing import Dict, Any

class Config:
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = Path("/workspace/output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_CACHE_DIR = Path("/workspace/model_cache")  # RunPod path'i
    LOG_DIR = Path("/workspace/output/logs")  # RunPod path'i
    
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        MODEL_CACHE_DIR.mkdir(exist_ok=True)
        LOG_DIR.mkdir(exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directories: {str(e)}")

    # API Configuration
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8001"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Model Configuration
    MODEL_CONFIG: Dict[str, Any] = {
        "model_id": os.getenv("MODEL_ID", "Lightricks/LTX-Video"),
        "use_auth_token": os.getenv("HF_TOKEN", None),
        "torch_dtype": "float16",  # or "float32" for CPU
        "low_cpu_mem_usage": True,
        "cache_dir": "/workspace/model_cache"  # Model cache dizinini açıkça belirt
    }
    
    # Generation Parameters
    MAX_FRAMES = int(os.getenv("MAX_FRAMES", "32"))
    MAX_RESOLUTION = int(os.getenv("MAX_RESOLUTION", "1024"))
    DEFAULT_FPS = int(os.getenv("DEFAULT_FPS", "8"))
    
    # Performance Settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    USE_MIXED_PRECISION = os.getenv("USE_MIXED_PRECISION", "true").lower() == "true"
    ENABLE_VAE_SLICING = os.getenv("ENABLE_VAE_SLICING", "true").lower() == "true"
    MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "300"))

    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))


    # Queue Settings
    QUEUE_NAME = os.getenv("QUEUE_NAME", "video_generation_queue")
    QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "1000"))
    QUEUE_MAX_PRIORITY = int(os.getenv("QUEUE_MAX_PRIORITY", "10"))
    QUEUE_MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "5"))
    QUEUE_RETRY_DELAY = int(os.getenv("QUEUE_RETRY_DELAY", "2"))
    QUEUE_PREFETCH_COUNT = int(os.getenv("QUEUE_PREFETCH_COUNT", "1"))

    # Resource Thresholds
    MEMORY_THRESHOLD = float(os.getenv("MEMORY_THRESHOLD", "90.0"))
    GPU_MEMORY_THRESHOLD = float(os.getenv("GPU_MEMORY_THRESHOLD", "90.0"))
    CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "90.0"))
    CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3600"))

    # Rate Limiting
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))

    # Cleanup Settings
    FILE_CLEANUP_AGE = int(os.getenv("FILE_CLEANUP_AGE", "86400"))
    COMPLETED_JOBS_MAX_AGE = int(os.getenv("COMPLETED_JOBS_MAX_AGE", "3600"))

    # Error Handling
    MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "3"))
    ERROR_COOLDOWN_PERIOD = int(os.getenv("ERROR_COOLDOWN_PERIOD", "300"))

    # Monitoring and Metrics
    ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Genişletilmiş config validasyonu"""
        try:
            # Resource limits
            assert 0 <= cls.MEMORY_THRESHOLD <= 100, "Memory threshold must be between 0 and 100"
            assert 0 <= cls.GPU_MEMORY_THRESHOLD <= 100, "GPU memory threshold must be between 0 and 100"
            assert 0 <= cls.CPU_THRESHOLD <= 100, "CPU threshold must be between 0 and 100"
            
            # Queue settings
            assert cls.QUEUE_MAX_RETRIES > 0, "Max retries must be positive"
            assert cls.QUEUE_RETRY_DELAY > 0, "Retry delay must be positive"
            
            # Generation limits
            assert cls.MAX_FRAMES > 0, "Max frames must be positive"
            assert cls.MAX_RESOLUTION > 0, "Max resolution must be positive"
            assert cls.BATCH_SIZE > 0, "Batch size must be positive"
            
            # Timeouts and intervals
            assert cls.MODEL_TIMEOUT > 0, "Model timeout must be positive"
            assert cls.CLEANUP_INTERVAL > 0, "Cleanup interval must be positive"
            
            # Directory validation
            assert cls.OUTPUT_DIR.exists(), "Output directory does not exist"
            assert cls.MODEL_CACHE_DIR.exists(), "Model cache directory does not exist"
            assert cls.LOG_DIR.exists(), "Log directory does not exist"
            
            return True
            
        except AssertionError as e:
            raise AssertionError(f"Configuration validation failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during configuration validation: {str(e)}")

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Model konfigürasyonunu döndür"""
        return cls.MODEL_CONFIG.copy()