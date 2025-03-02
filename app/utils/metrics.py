import psutil
import torch
from typing import Dict

class SystemMetrics:
    @staticmethod
    def get_metrics() -> Dict:
        metrics = {
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent()
        }
        
        if torch.cuda.is_available():
            metrics["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "percentage": torch.cuda.memory_allocated() / torch.cuda.memory_reserved() * 100
            }
            
        return metrics