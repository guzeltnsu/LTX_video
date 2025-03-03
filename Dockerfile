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

# Pip'i güncelle
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# SciPy önce yükle (kritik bağımlılık)
RUN pip install --no-cache-dir scipy==1.11.4

# Paketleri kademeli olarak yükle (kritik paketleri önce)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2
RUN pip install --no-cache-dir numpy==1.26.0 Pillow>=10.2.0
RUN pip install --no-cache-dir huggingface_hub==0.17.3
RUN pip install --no-cache-dir tokenizers==0.13.3
RUN pip install --no-cache-dir safetensors==0.4.0
RUN pip install --no-cache-dir transformers==4.30.2

# Diffusers paketini kararlı sürüm yerine GitHub ana dalından yükle
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

RUN pip install --no-cache-dir accelerate==0.20.3

# Diğer paketleri yükle (ML ekosistemi dışındakiler)
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

# Kurulum doğrulaması
RUN python -c "import huggingface_hub, diffusers, transformers, scipy; \
    print(f'huggingface_hub: {huggingface_hub.__version__}, diffusers: {diffusers.__version__}, transformers: {transformers.__version__}, scipy: {scipy.__version__}')"

# Üretim ortamı (devamı aynı şekilde)
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
COPY logging.conf /workspace/

# Çevre değişkenlerini ayarla
ENV PYTHONPATH="/workspace" \
    PYTHONUNBUFFERED=1 \
    API_HOST="0.0.0.0" \
    API_PORT="8001" \
    DEBUG_MODE="false" \
    HF_HOME="/workspace/model_cache" \
    MODE="serverless" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=all

# Port aç
EXPOSE 8001

# Başlatma betik dosyasını oluştur
RUN echo '#!/bin/bash\n\
set -x\n\
echo "MODE: $MODE"\n\
echo "Python versiyonu:"\n\
python --version\n\
echo "PYTHONPATH: $PYTHONPATH"\n\
echo "HuggingFace Hub versiyonu:"\n\
python -c "import huggingface_hub; print(huggingface_hub.__version__)"\n\
echo "Diffusers versiyonu:"\n\
python -c "import diffusers; print(diffusers.__version__)"\n\
echo "Transformers versiyonu:"\n\
python -c "import transformers; print(transformers.__version__)"\n\
echo "SciPy versiyonu:"\n\
python -c "import scipy; print(scipy.__version__)"\n\
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