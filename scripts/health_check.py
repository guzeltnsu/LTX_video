# scripts/health_check.py

import requests
import sys
import os
import time

def check_api_health():
    """API sağlık kontrolü için basit bir HTTP isteği yap"""
    try:
        # API modunda çalışıyorsa
        response = requests.get(f"http://localhost:8001/health")
        return response.status_code == 200
    except:
        # Serverless modda veya henüz başlamadıysa
        # Process varlığını kontrol et
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if any('python' in cmd for cmd in proc.info['cmdline'] if cmd):
                if any('/workspace/handler.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                    return True
        return False

if __name__ == "__main__":
    max_attempts = 5
    for attempt in range(max_attempts):
        if check_api_health():
            print("Service is healthy!")
            sys.exit(0)
        print(f"Health check attempt {attempt+1}/{max_attempts} failed, retrying...")
        time.sleep(2)
    
    print("Service is not healthy after maximum attempts")
    sys.exit(1)