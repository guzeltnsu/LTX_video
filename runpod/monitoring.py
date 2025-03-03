# runpod/monitoring.py

import time
import psutil
import os
from prometheus_client import start_http_server, Gauge, Counter
import threading
import logging

logger = logging.getLogger("monitoring")

# Metrikler
gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage')
gpu_memory = Gauge('gpu_memory_used_mb', 'GPU memory usage in MB')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
job_counter = Counter('processed_jobs_total', 'Total number of processed jobs')
error_counter = Counter('job_errors_total', 'Total number of job errors')

def collect_system_metrics():
    """Sistem metriklerini topla ve prometheus metriklerine aktar"""
    while True:
        try:
            # CPU kullanımı
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.set(cpu_percent)
            
            # Bellek kullanımı
            memory = psutil.virtual_memory()
            memory_usage.set(memory.percent)
            
            # GPU metrikleri (nvidia-smi aracılığıyla)
            try:
                import subprocess
                gpu_info = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']
                ).decode('utf-8').strip().split('\n')
                
                for i, line in enumerate(gpu_info):
                    util, mem = line.split(',')
                    gpu_usage.labels(gpu=i).set(float(util))
                    gpu_memory.labels(gpu=i).set(float(mem))
            except:
                logger.warning("GPU metrikleri alınamadı")
                
        except Exception as e:
            logger.error(f"Metrik toplama hatası: {e}")
        
        time.sleep(15)  # Her 15 saniyede bir güncelle

def start_monitoring(port=8000):
    """Prometheus metrik sunucusunu başlat ve metrik toplamayı başlat"""
    try:
        # Prometheus metrik sunucusunu başlat
        start_http_server(port)
        logger.info(f"Prometheus metrik sunucusu port {port} üzerinde başlatıldı")
        
        # Arka planda metrik toplamayı başlat
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()
        logger.info("Sistem metriklerini toplama başlatıldı")
        
        return True
    except Exception as e:
        logger.error(f"İzleme başlatma hatası: {e}")
        return False

if __name__ == "__main__":
    # Direkt çalıştırıldığında metrik sunucusunu başlat
    logging.basicConfig(level=logging.INFO)
    start_monitoring()
    
    # Ana thread'i çalışır durumda tut
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("İzleme sonlandırılıyor...")