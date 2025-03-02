import pytest
import time
import torch
import numpy as np
from pathlib import Path
from app.services.video_generator import VideoGenerator
from app.models.params import VideoGenerationParams, VideoPreset, VideoFormat
import asyncio

@pytest.mark.skip(reason="Uzun süren bir test")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA gerektirir")
@pytest.mark.parametrize("preset", [
    VideoPreset.MOBILE,
    VideoPreset.HD
])
@pytest.mark.parametrize("steps", [
    10, 20, 50
])
@pytest.mark.asyncio
async def test_generation_performance(preset, steps):
    """Farklı presetler ve adım sayıları ile performans testi"""
    generator = VideoGenerator()
    if not generator.model_ready:
        generator.load_model()
    
    # Parametreler
    params = VideoGenerationParams(
        prompt="A beautiful landscape with mountains and rivers",
        preset=preset,
        format=VideoFormat.MP4,
        num_frames=16,
        num_inference_steps=steps
    )
    
    # Önceki VRAM kullanımı
    before_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    # Zamanlamayı başlat
    start_time = time.time()
    
    # Video oluştur
    success, _, _ = await generator.generate_video(params, f"perf_test_{preset}_{steps}")
    
    # Tamamlanma süresi
    generation_time = time.time() - start_time
    
    # Sonraki VRAM kullanımı
    after_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    # Bellek sızıntısı kontrolü
    await generator.cleanup_resources()
    final_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    print(f"\nPerformance - Preset: {preset}, Steps: {steps}")
    print(f"  Generation Time: {generation_time:.2f} seconds")
    print(f"  VRAM Usage: Before={before_vram:.2f}GB, Peak={after_vram:.2f}GB, After Cleanup={final_vram:.2f}GB")
    print(f"  Memory Overhead: {after_vram-before_vram:.2f}GB")
    print(f"  Memory Leak: {final_vram-before_vram:.2f}GB")
    
    # Başarı kontrolü
    assert success is True
    
    # Bellek sızıntısı olmamalı (0.1 GB tolerans)
    assert final_vram <= before_vram + 0.1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA gerektirir")
def test_memory_cleanup_efficiency():
    """Bellek temizleme performansı testi"""
    generator = VideoGenerator()
    if not generator.model_ready:
        generator.load_model()
    
    # VRAM kullanımını ölç
    base_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    # Büyük bir tensör oluşturup belleği kullan
    dummy_tensor = torch.ones((4096, 4096), device="cuda")
    peak_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    # Dummy tensör'ü GC'ye bırak
    del dummy_tensor
    
    # Zamanlamayı başlat
    start_time = time.time()
    
    # Temizlik işlemi
    asyncio.run(generator.cleanup_resources())
    
    # Temizlik süresi
    cleanup_time = time.time() - start_time
    
    # Temizlik sonrası VRAM
    final_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    print(f"\nMemory Cleanup Performance")
    print(f"  Base VRAM: {base_vram:.2f}GB")
    print(f"  Peak VRAM: {peak_vram:.2f}GB")
    print(f"  Final VRAM: {final_vram:.2f}GB")
    print(f"  Cleanup Time: {cleanup_time:.4f} seconds")
    print(f"  Recovery Percentage: {((peak_vram - final_vram) / (peak_vram - base_vram)) * 100:.1f}%")
    
    # Temizlik en az %90 etkin olmalı
    cleanup_efficiency = (peak_vram - final_vram) / (peak_vram - base_vram)
    assert cleanup_efficiency >= 0.9
    
    # Temizlik 1 saniyeden az sürmeli
    assert cleanup_time < 1.0