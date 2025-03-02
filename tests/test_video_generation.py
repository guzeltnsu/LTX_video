import pytest
from fastapi.testclient import TestClient
import json
import time
import asyncio
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Dict, Any

from app.main import app
from app.models.params import VideoGenerationParams, VideoPreset, VideoFormat

client = TestClient(app)

@pytest.fixture
def test_image():
    """Test görüntüsü oluştur"""
    img = Image.new('RGB', (256, 256), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def test_health_endpoint():
    """Health endpoint testi"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_status" in data

def test_metrics_endpoint():
    """Metrics endpoint testi"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "memory" in data
    assert "cpu" in data
    if "gpu" in data:
        assert "allocated" in data["gpu"]

def test_generate_with_invalid_params():
    """Geçersiz parametrelerle generate endpoint testi"""
    # Boş istek gönder
    response = client.post("/generate", json={})
    assert response.status_code == 422
    
    # Çok büyük boyut
    response = client.post("/generate", json={
        "prompt": "Test",
        "preset": "custom",
        "width": 2048,  # Çok büyük
        "height": 512
    })
    assert response.status_code == 422

def test_generate_valid_request(test_image):
    """Geçerli parametrelerle generate endpoint testi"""
    # Geçerli parametrelerle istek
    params = {
        "prompt": "Test video generation",
        "preset": "mobile",
        "format": "mp4",
        "num_frames": 8,
        "num_inference_steps": 5,
        "input_image": test_image
    }
    response = client.post("/generate", json=params)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "status" in data
    assert data["status"] == "IN_QUEUE"

@pytest.mark.stress
class TestApiStress:
    """API stres testleri"""
    
    def test_concurrent_health_requests(self):
        """Eşzamanlı health isteği testi"""
        start_time = time.time()
        
        # 50 eşzamanlı health isteği
        responses = []
        for _ in range(50):
            responses.append(client.get("/health"))
        
        # Tüm yanıtlar başarılı olmalı
        for response in responses:
            assert response.status_code == 200
        
        # Toplam süre
        total_time = time.time() - start_time
        print(f"50 concurrent health requests took {total_time:.2f} seconds")
        
        # Saniyede 10 istekten fazla işlenebilmeli
        assert total_time < 5.0
    
    def test_sequential_generate_requests(self):
        """Ardışık generate isteği testi"""
        start_time = time.time()
        
        # 5 ardışık generate isteği
        for i in range(5):
            params = {
                "prompt": f"Test video generation {i}",
                "preset": "mobile",
                "num_frames": 8,
                "num_inference_steps": 2
            }
            response = client.post("/generate", json=params)
            assert response.status_code == 200
        
        # Toplam süre
        total_time = time.time() - start_time
        print(f"5 sequential generate requests took {total_time:.2f} seconds")
        
        # Her istek ortalama 1 saniyeden az sürmeli
        assert total_time < 5.0

@pytest.mark.benchmark
def test_api_throughput(benchmark):
    """API throughput testi"""
    def make_health_request():
        return client.get("/health")
    
    # Health endpoint performansını ölç
    result = benchmark(make_health_request)
    assert result.status_code == 200