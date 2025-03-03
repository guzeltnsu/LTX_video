import asyncio
from diffusers import LTXImageToVideoPipeline
import torch
import numpy as np
from pathlib import Path
import logging
import io
import os
import base64
from PIL import Image
import psutil
from typing import Tuple, Optional
import cv2

from app.config import Config
from app.models.params import VideoGenerationParams
from app.services.queue_manager import QueueManager

logger = logging.getLogger(__name__)

class VideoGenerator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.model_ready = False
        self.queue_manager = QueueManager()
        self.load_model()

    async def cleanup_resources(self):
        """Sistem kaynaklarını temizle"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Model'i CPU'ya taşı
            if self.pipe is not None and self.device.type == "cuda":
                self.pipe.unet.to("cpu")
                torch.cuda.empty_cache()
                # İşlem bittikten sonra geri taşı
                self.pipe.unet.to(self.device)
            
            # Manuel GC başlat
            import gc
            gc.collect()
            
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")

    def check_resources(self) -> bool:
        try:
            memory = psutil.virtual_memory()
            if memory.percent > Config.MEMORY_THRESHOLD:
                logger.warning(f"High memory usage: {memory.percent}%")
                asyncio.create_task(self.cleanup_resources())
                return False

            if torch.cuda.is_available():
                # Daha detaylı GPU memory kontrolü
                allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                if allocated/reserved > 0.9 or allocated > Config.GPU_MEMORY_THRESHOLD:
                    logger.warning(f"High GPU memory usage: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                    asyncio.create_task(self.cleanup_resources())
                    return False

            return True
        except Exception as e:
            logger.error(f"Resource check error: {str(e)}")
            return False

    def load_model(self):
        try:
            model_config = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "balanced",  
                "use_safetensors": True,  # Daha hızlı model yükleme
                "low_cpu_mem_usage": True
            }
            
            # SentencePiece hatasını önlemek için
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            self.pipe = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                **model_config,
                cache_dir="/workspace/model_cache"  # RunPod path'i
            ).to(self.device)
            
            if self.device.type == "cuda":
                self.pipe.enable_vae_slicing()  # VAE bellek kullanımını azaltır
                self.pipe.enable_model_cpu_offload()  # Bellek tasarrufu sağlar
            
            self.model_ready = True
            logger.info(f"Model loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.pipe = None
            self.model_ready = False
            return False

    def process_input_image(self, image_base64: str) -> Optional[Image.Image]:
        """Base64 kodlu görüntüyü işle ve PIL.Image olarak döndür"""
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_data))
            return img
        except Exception as e:
            logger.error(f"Error processing input image: {str(e)}")
            return None

    def save_video(self, frames, output_path: Path, fps: int) -> None:
        try:
            # İlk frame'den video writer'ı yapılandır
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            
            # RunPod için output path'i kontrol et
            output_path = Path("/workspace/output") / output_path.name
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Frame'leri işle
            for frame in frames:
                # Modelden gelen frame'leri [0,1] -> [0,255] aralığına dönüştür
                frame = (frame * 255).astype(np.uint8)
                # RGB -> BGR dönüşümü (OpenCV BGR kullanır)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            # Video writer'ı kapat
            out.release()
            logger.info(f"Video saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            raise

    def _optimize_generation_params(self, params: VideoGenerationParams):
        """Parametre optimizasyonu"""
        # Parametreleri kopyala
        optimized = params.model_copy(deep=True)
        
        # Kullanılmayan parametreleri temizle
        generation_params = optimized.model_dump(exclude_unset=True)
        
        # FPS değerini kontrol et
        if not optimized.fps or optimized.fps < 1:
            optimized.fps = 8  # Varsayılan değer
            
        # Boyut kontrolü
        if optimized.width % 8 != 0:
            optimized.width = (optimized.width // 8) * 8
        if optimized.height % 8 != 0:
            optimized.height = (optimized.height // 8) * 8
            
        # Frame sayısı kontrolü
        if optimized.num_frames > 32:
            optimized.num_frames = 32  # Maksimum değer
            
        return optimized

    async def generate_video(self, params, job_id: str) -> Tuple[bool, str, str]:
        try:
            if not self.check_resources():
                return False, "", "Insufficient resources"

            # Model parametrelerini optimize et
            optimized_params = self._optimize_generation_params(params)
        
            # Generation başlamadan önce cache temizliği
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Pipeline için parametreleri hazırla
            generation_kwargs = {
                "prompt": optimized_params.prompt,
                "negative_prompt": optimized_params.negative_prompt,
                "num_frames": optimized_params.num_frames,
                "num_inference_steps": optimized_params.num_inference_steps,
                "guidance_scale": optimized_params.guidance_scale,
                "height": optimized_params.height,
                "width": optimized_params.width,
            }
            
            # İnput image varsa ekle
            if optimized_params.input_image:
                input_image = self.process_input_image(optimized_params.input_image)
                if input_image:
                    generation_kwargs["image"] = input_image
                else:
                    return False, "", "Invalid input image"
            
            # Seed belirtilmişse, generator ekle
            if optimized_params.seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(optimized_params.seed)
                generation_kwargs["generator"] = generator

            # Bellek temizliği için callback fonksiyonu
            generation_steps = optimized_params.num_inference_steps
            memory_cleanup_frequency = max(1, generation_steps // 4)
            
            def cleanup_callback(step, timestep, callback_kwargs):
                if step > 0 and step % memory_cleanup_frequency == 0:
                    torch.cuda.empty_cache()
                return callback_kwargs

            with torch.inference_mode(), torch.cuda.amp.autocast():  # Mixed precision
                output = self.pipe(
                    **generation_kwargs,
                    callback=cleanup_callback,
                    callback_steps=1
                )

                if "videos" not in output or not output["videos"]:
                    return False, "", "Generation failed"

                video_path = Config.OUTPUT_DIR / f"{job_id}.mp4"
                saved_path = await asyncio.to_thread(
                    self.save_video, 
                    output["videos"][0], 
                    video_path, 
                    optimized_params.fps
                )
                
                # Son bellek temizliği
                await self.cleanup_resources()
            
                return True, str(saved_path), ""

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            torch.cuda.empty_cache()  # Hata durumunda memory temizliği
            import gc
            gc.collect()  # Garbage collection başlat
            return False, "", str(e)