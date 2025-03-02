from setuptools import setup, find_packages

setup(
    name="app",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.10,<4.0',
    install_requires=[
        'torch>=2.1.2',
        'torchvision>=0.16.2',
        'diffusers==0.32.2',
        'transformers==4.48.2',
        'huggingface-hub==0.28.1',  # Tüm paketlerle uyumlu versiyon aralığı
        'tokenizers==0.21.0',
        'fastapi>=0.109.0',
        'uvicorn>=0.25.0',
        'runpod>=1.0.0',
        'numpy>=1.26.0',
        'pillow>=10.0.0',
        'python-multipart>=0.0.9',
        'opencv-python>=4.7.0',
        'psutil>=5.9.0',
    ],
)