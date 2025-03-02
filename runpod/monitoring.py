import psutil
import time
import torch
from datetime import datetime

class Monitor:
    def __init__(self):
        self.start_time = time.time()
        
    def log_event(self, event_type, message, extra_data=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}][{event_type}] {message}")
        if extra_data:
            print(f"Extra data: {extra_data}")
            
    def get_system_metrics(self):
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "uptime_seconds": time.time() - self.start_time
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics.update({
                "device": "cuda",
                "gpu_memory_used_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_utilization": None  # NVIDIA-SMI gerekiyor
            })
        else:
            metrics["device"] = "cpu"
            
        return metrics
        
    def log_metrics(self):
        metrics = self.get_system_metrics()
        self.log_event("metrics", "System metrics", metrics)

monitor = Monitor()