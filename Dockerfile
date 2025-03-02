# Temel imaj olarak PyTorch ile CUDA 12.1 destekli imajı kullan
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

# Geliştirme ortamı
FROM base AS builder
WORKDIR /build

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sanal ortam oluştur
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Önce numpy 1.x sürümünü yükle (NumPy 2.x uyumsuzluğunu önlemek için)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.26.0

# Paketleri belirli sürümlerde yükle (huggingface_hub'ı öncelikle yükle)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 && \
    pip install --no-cache-dir huggingface_hub==0.17.3 && \
    pip install --no-cache-dir tokenizers==0.14.1 && \
    pip install --no-cache-dir safetensors==0.4.0 && \
    pip install --no-cache-dir transformers==4.36.2 && \
    pip install --no-cache-dir diffusers==0.23.1 && \
    pip install --no-cache-dir accelerate==0.25.0

# Diğer gereksinimleri yükle
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.25.0 \
    python-multipart==0.0.9 \
    pydantic==2.6.1 \
    opencv-python-headless==4.8.1.78 \
    imageio==2.33.0 \
    psutil==5.9.8 \
    prometheus_client>=0.19.0 \
    loguru==0.7.2 \
    bitsandbytes==0.41.3 \
    runpod==1.3.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    python-json-logger>=2.0.7

# Üretim ortamı
FROM base AS runtime
WORKDIR /workspace

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sanal ortamı kopyala
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Çalışma dizinlerini oluştur
RUN mkdir -p /workspace/output /workspace/model_cache /workspace/output/logs

# Uygulama dosyalarını kopyala
COPY app/ /workspace/app/
COPY handler.py /workspace/
COPY runpod/ /workspace/runpod/

# Çevre değişkenlerini ayarla
ENV PYTHONPATH="/workspace" \
    PYTHONUNBUFFERED=1 \
    API_HOST="0.0.0.0" \
    API_PORT="8001" \
    DEBUG_MODE="false" \
    HF_HOME="/workspace/model_cache" \
    MODE="serverless"

# Port aç
EXPOSE 8001

# Başlatma betik dosyasını oluştur
RUN echo '#!/bin/bash\n\
set -x\n\
echo "MODE: $MODE"\n\
echo "Python versiyonu:"\n\
python --version\n\
echo "NumPy versiyonu:"\n\
python -c "import numpy; print(numpy.__version__)"\n\
echo "HuggingFace Hub versiyonu:"\n\
python -c "import huggingface_hub; print(huggingface_hub.__version__)"\n\
if [ "$MODE" = "api" ]; then\n\
  echo "API modu başlatılıyor..."\n\
  python -m app.main\n\
elif [ "$MODE" = "serverless" ]; then\n\
  echo "Serverless modu başlatılıyor..."\n\
  python /workspace/handler.py\n\
else\n\
  echo "Geçersiz MODE: $MODE"\n\
  exit 1\n\
fi' > /workspace/start.sh && chmod +x /workspace/start.sh

# Başlatma betiğini çalıştır
CMD ["/workspace/start.sh"]