from pydantic import BaseModel, Field, field_validator
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
import io
from enum import Enum

class VideoPreset(str, Enum):   
    MOBİLE = "mobile"
    HD = "hd"
    FULL_HD = "full_hd"
    ULTRA_HD = "ultra_hd"
    CUSTOM = "custom"

class VideoFormat(str, Enum):
    MP4 = "mp4"
    GIF = "gif"
    WEBM = "webm"

class VideoGenerationParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    preset: VideoPreset = Field(VideoPreset.HD, description="Predefined video settings")
    format: VideoFormat = Field(VideoFormat.MP4, description="Output video format")
    num_frames: int = Field(8, ge=1, le=120)
    fps: Optional[int] = Field(None, ge=1, le=60)
    num_inference_steps: int = Field(20, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    width: Optional[int] = Field(None, ge=256, le=1024)
    height: Optional[int] = Field(None, ge=256, le=1024)
    seed: Optional[int] = None
    priority: int = Field(0, ge=0, le=9)
    input_image: Optional[str] = None
    
    @field_validator('input_image')
    @classmethod
    def validate_input_image(cls, v):
        if v:
            try:
                image_data = base64.b64decode(v)
                img = Image.open(io.BytesIO(image_data))
                if img.size[0] * img.size[1] > 1024 * 1024:
                    raise ValueError("Image too large")
                if img.format not in ['JPEG', 'PNG']:
                    raise ValueError("Unsupported format")
                return v
            except Exception as e:
                raise ValueError(f"Invalid image: {str(e)}")
        return v
    
    @field_validator('width', 'height', 'fps', mode='before')
    @classmethod
    def apply_preset(cls, v, info):
        # Eğer değer zaten ayarlanmışsa ve preset CUSTOM ise, değeri koru
        if v is not None and info.data.get('preset') == VideoPreset.CUSTOM:
            return v
            
        # Değer ayarlanmamışsa ve diğer presetlerden biri seçilmişse, preset değerlerini kullan
        preset = info.data.get('preset')
        field_name = info.field_name
        
        preset_values = {
            VideoPreset.MOBILE: {"width": 256, "height": 256, "fps": 8},
            VideoPreset.HD: {"width": 512, "height": 512, "fps": 15},
            VideoPreset.FULL_HD: {"width": 768, "height": 768, "fps": 24},
            VideoPreset.ULTRA_HD: {"width": 1024, "height": 1024, "fps": 30},
        }
        
        if preset in preset_values:
            return preset_values[preset].get(field_name, v)
        return v

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    queue_position: Optional[int] = None
    system_metrics: Optional[dict] = None