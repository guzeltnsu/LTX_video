import runpod
import torch
from diffusers import DiffusionPipeline
import numpy as np
import cv2
import base64
from pathlib import Path
import time
import os
import io
from PIL import Image

class Monitor:
    def __init__(self):
        self.start_time = time.time()
        
    def log_event(self, event_type, message, extra_data=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        print(f"[{timestamp}][{event_type}] {message}")
        if extra_data:
            print(f"Extra data: {extra_data}")
            
    def get_system_metrics(self):
        metrics = {
            "uptime_seconds": time.time() - self.start_time
        }
        if torch.cuda.is_available():
            metrics.update({
                "device": "cuda",
                "gpu_memory_used_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
            })
        else:
            metrics["device"] = "cpu"
        return metrics
        
    def log_metrics(self):
        metrics = self.get_system_metrics()
        self.log_event("metrics", "System metrics", metrics)

monitor = Monitor()

def check_directories():
    """Gerekli dizinleri kontrol edip oluşturur."""
    directories = {
        "/workspace/output": "Output directory",
        "/workspace/model_cache": "Model cache directory",
        "/workspace/output/logs": "Log directory"
    }
    
    for path, description in directories.items():
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                print(f"Warning: {description} ({path}) is not writable")
            else:
                print(f"✓ {description} ({path}) is ready")
        except Exception as e:
            print(f"Error: {description} ({path}) could not be created: {str(e)}")
            return False
    return True

if not check_directories():
    print("Critical directories could not be created. Worker cannot start.")
    exit(1)
    
class ModelHandler:
    def __init__(self):
        monitor.log_event("initialization", "Starting model initialization")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        monitor.log_event("device", f"Using device: {self.device}")
        self.model = self._load_model()
        monitor.log_event("initialization", "Model loaded successfully")

    def _load_model(self):
        try:
            # DiffusionPipeline model ID'ye bakarak doğru pipeline'ı otomatik yükler
            model = DiffusionPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                cache_dir="/workspace/model_cache"
            )
            
            if self.device == "cuda":
                model.enable_vae_slicing()
                model.enable_model_cpu_offload()
            
            return model
        except Exception as e:
            monitor.log_event("error", f"Model load failed: {str(e)}")
            raise

    def generate(self, params):
        try:
            start_time = time.time()
            monitor.log_event("generation", "Starting video generation", params)
            
            # Eğer input_image sağlanmışsa, base64 çözme işlemi yapılıyor.
            if "input_image" in params and params["input_image"]:
                try:
                    image_data = base64.b64decode(params["input_image"])
                    image = Image.open(io.BytesIO(image_data))
                    params["image"] = image
                    del params["input_image"]
                except Exception as e:
                    monitor.log_event("error", f"Image processing failed: {str(e)}")
                    raise ValueError(f"Invalid input image: {str(e)}")
            
            with torch.inference_mode():
                torch.cuda.empty_cache()
                output = self.model(**params)
                generation_time = time.time() - start_time
                monitor.log_event("generation", f"Generation completed in {generation_time:.2f} seconds")
                return output
        except Exception as e:
            monitor.log_event("error", f"Generation failed: {str(e)}")
            torch.cuda.empty_cache()
            raise

def save_video(video_frames, fps=8):
    try:
        output_path = Path("/workspace/output/output.mp4")
        video_frames = np.stack([np.array(frame) for frame in video_frames])
        height, width = video_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in video_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        monitor.log_event("video", f"Video saved to {output_path}")
        return output_path
    except Exception as e:
        monitor.log_event("error", f"Video save failed: {str(e)}")
        raise

def encode_video_to_base64(video_path):
    try:
        with open(video_path, "rb") as video_file:
            encoded = base64.b64encode(video_file.read()).decode('utf-8')
            monitor.log_event("video", "Video encoded to base64")
            return encoded
    except Exception as e:
        monitor.log_event("error", f"Video encoding failed: {str(e)}")
        raise

def handler(event):
    try:
        start_time = time.time()
        monitor.log_metrics()  # Başlangıç sistem metrikleri
        
        input_data = event.get("input", {})
        
        # Gerekli alan kontrolü: prompt veya input_image en azından bir tanesi olmalı.
        if not "prompt" in input_data and not "input_image" in input_data:
            return {"error": "Either prompt or input_image is required"}
        
        params = {
            "prompt": input_data.get("prompt", None),
            "negative_prompt": input_data.get("negative_prompt", None),
            "num_frames": input_data.get("num_frames", 16),
            "fps": input_data.get("fps", 8),
            "num_inference_steps": input_data.get("num_inference_steps", 20),
            "guidance_scale": input_data.get("guidance_scale", 7.5),
            "width": input_data.get("width", 512),
            "height": input_data.get("height", 512),
            "input_image": input_data.get("input_image", None)
        }
        
        if "seed" in input_data and input_data["seed"] is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(input_data["seed"])
            params["generator"] = generator
        
        model_handler = ModelHandler()
        output = model_handler.generate(params)
        
        if "videos" not in output or not output["videos"]:
            return {"error": "Video generation failed"}
            
        video_path = save_video(output["videos"][0], params.get("fps", 8))
        video_base64 = encode_video_to_base64(video_path)
        
        # Geçici dosyaların temizlenmesi
        if video_path.exists():
            try:
                video_path.unlink()
                monitor.log_event("cleanup", "Temporary files cleaned")
            except Exception as e:
                monitor.log_event("warning", f"Failed to clean up temporary files: {str(e)}")
        
        total_time = time.time() - start_time
        monitor.log_metrics()  # Son metrikler
        
        return {
            "video": video_base64,
            "status": "success",
            "metrics": {
                "total_time": total_time,
                "system_metrics": monitor.get_system_metrics()
            }
        }
    except Exception as e:
        monitor.log_event("error", f"Handler error: {str(e)}")
        torch.cuda.empty_cache()
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })