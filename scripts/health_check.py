import requests
import sys
import time
from typing import Dict, Optional

def check_health(url: str, max_retries: int = 5, retry_delay: int = 5) -> Optional[Dict]:
    """
    Check the health of the API service
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                return response.json()
            print(f"Attempt {attempt + 1}/{max_retries}: Service unhealthy")
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return None

def main():
    api_url = "http://localhost:8001"
    health_info = check_health(api_url)
    
    if health_info:
        print("Service Health Information:")
        print(f"Status: {health_info.get('status', 'unknown')}")
        print(f"Model Status: {health_info.get('model_status', 'unknown')}")
        print(f"Active Jobs: {health_info.get('active_jobs', 0)}")
        print(f"Queue Length: {health_info.get('queue_length', 0)}")
        
        system_metrics = health_info.get('system_metrics', {})
        if system_metrics:
            print("\nSystem Metrics:")
            print(f"CPU Usage: {system_metrics.get('cpu_usage', 'N/A')}%")
            print(f"Memory Usage: {system_metrics.get('memory_usage', 'N/A')}%")
            if 'gpu_memory' in system_metrics:
                gpu = system_metrics['gpu_memory']
                print(f"GPU Memory: {gpu.get('allocated', 0) / 1024**3:.2f}GB / {gpu.get('reserved', 0) / 1024**3:.2f}GB")
        
        sys.exit(0)
    else:
        print("Service is unhealthy")
        sys.exit(1)

if __name__ == "__main__":
    main()