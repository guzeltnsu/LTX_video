from enum import Enum
import logging
import traceback
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    INPUT_VALIDATION = "input_validation"
    MODEL_LOADING = "model_loading"
    RESOURCE_CONSTRAINT = "resource_constraint"
    GENERATION_FAILURE = "generation_failure"
    BACKEND_ERROR = "backend_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"

class AppError(Exception):
    def __init__(self, 
                 message: str, 
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.category = category
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

def handle_error(func):
    """Hata işleme decorator'ı"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AppError as e:
            # Yapılandırılmış uygulama hatası
            logger.error(f"Application error: {e.message}", 
                         extra={
                             "category": e.category.value,
                             "error_code": e.error_code,
                             "details": e.details
                         })
            return {
                "error": {
                    "message": e.message,
                    "category": e.category.value,
                    "code": e.error_code,
                    "details": e.details
                }
            }
        except Exception as e:
            # Yakalanmamış genel hata
            error_details = {
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"Unhandled exception: {str(e)}", 
                         extra={"traceback": traceback.format_exc()})
            return {
                "error": {
                    "message": "An unexpected error occurred",
                    "category": ErrorCategory.UNKNOWN.value,
                    "details": error_details
                }
            }
    return wrapper

def map_error(e: Exception) -> Tuple[ErrorCategory, str, Dict[str, Any]]:
    """
    Hata türüne göre kategori ve detayları belirler
    """
    import torch
    
    if isinstance(e, ValueError) and "input" in str(e).lower():
        return ErrorCategory.INPUT_VALIDATION, str(e), {}
    elif isinstance(e, torch.cuda.OutOfMemoryError):
        return ErrorCategory.RESOURCE_CONSTRAINT, "GPU memory exceeded", {
            "suggestion": "Try reducing model parameters (resolution, steps)"
        }
    elif "model" in str(e).lower() and "load" in str(e).lower():
        return ErrorCategory.MODEL_LOADING, "Failed to load model", {
            "details": str(e)
        }
    elif "generation" in str(e).lower():
        return ErrorCategory.GENERATION_FAILURE, "Failed to generate video", {
            "details": str(e)
        }
    elif isinstance(e, ConnectionError):
        return ErrorCategory.NETWORK_ERROR, "Network connection error", {
            "details": str(e)
        }
    else:
        return ErrorCategory.UNKNOWN, str(e), {}