import logging
import time
import psutil
import json
from pathlib import Path
import os
from datetime import datetime
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TelemetryManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.start_time = time.time()
        self.log_dir = Path("/workspace/output/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.generation_times = []
        self.request_counts = 0
        self.error_counts = 0
        self.gpu_usage_samples = []
        
        # Telemetri dosyası
        self.telemetry_file = self.log_dir / f"telemetry_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        logger.info(f"Telemetry initialized, logging to {self.telemetry_file}")
    
    def log_request(self, endpoint: str, params: Dict[str, Any]):
        """API isteğini logla"""
        self.request_counts += 1
        
        # Hassas bilgileri kaldır
        safe_params = params.copy()
        if "input_image" in safe_params:
            safe_params["input_image"] = "[BASE64_IMAGE]"
        
        self._write_telemetry_entry({
            "type": "request",
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "params": safe_params
        })
    
    def log_generation(self, job_id: str, duration: float, params: Dict[str, Any], success: bool):
        """Video generation olayını logla"""
        self.generation_times.append(duration)
        
        if not success:
            self.error_counts += 1
        
        self._write_telemetry_entry({
            "type": "generation",
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id,
            "duration": duration,
            "success": success,
            "params": {k: v for k, v in params.items() if k != "input_image"}
        })
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Hata olayını logla"""
        self.error_counts += 1
        
        self._write_telemetry_entry({
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
            "details": details or {}
        })
    
    def log_system_metrics(self):
        """Sistem metriklerini örnekle ve logla"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - self.start_time,
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "cpu": {
                    "percent": psutil.cpu_percent(interval=0.1),
                    "count": psutil.cpu_count()
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "used": psutil.disk_usage("/").used,
                    "percent": psutil.disk_usage("/").percent
                }
            }
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_metrics = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated()
                }
                metrics["gpu"] = gpu_metrics
                
                # Sample for tracking
                self.gpu_usage_samples.append(gpu_metrics["allocated"] / (1024**3))  # GB
            
            self._write_telemetry_entry({
                "type": "system_metrics",
                "metrics": metrics
            })
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Performans özeti oluştur"""
        avg_generation_time = sum(self.generation_times) / max(1, len(self.generation_times))
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_counts,
            "error_count": self.error_counts,
            "error_rate": self.error_counts / max(1, self.request_counts),
            "avg_generation_time": avg_generation_time,
            "gpu_usage": {
                "current": self.gpu_usage_samples[-1] if self.gpu_usage_samples else 0,
                "average": sum(self.gpu_usage_samples) / max(1, len(self.gpu_usage_samples))
            } if torch.cuda.is_available() else {}
        }
    
    def _write_telemetry_entry(self, entry: Dict[str, Any]):
        """Telemetri kaydını dosyaya yaz"""
        try:
            with open(self.telemetry_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write telemetry: {e}")