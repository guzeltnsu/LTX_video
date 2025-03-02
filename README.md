# Let-Video API Service

A FastAPI-based service for generating videos using the Let-Video model from Lightricks Studio, optimized for RunPod serverless deployment.

## Features
- Image-to-Video generation using Let-Video model
- Serverless job processing with RunPod
- GPU acceleration with CUDA optimization
- Resource management and efficient memory usage
- Health monitoring and metrics

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support
- Docker
- RunPod account

## Quick Start
1. Clone the repository:
```bash
git clone <repository-url>
cd video-processing-api

2. Build and start the services:
```bash
docker-compose up --build
```

3. The API will be available at `http://localhost:8001`

## API Endpoints

- `POST /generate` - Create a new video generation job
- `GET /jobs/{job_id}` - Get job status
- `GET /health` - Check service health
- `GET /metrics` - Get system metrics

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install development dependencies:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

3. Run tests:
```bash
pytest tests/
```

## Deployment

For production deployment:

1. Update environment variables in docker-compose.yml
2. Configure GPU resources as needed
3. Deploy using docker-compose:
```bash
docker-compose -f docker-compose.yml up -d
```

## Monitoring

- RabbitMQ management interface: http://localhost:15672
- API metrics endpoint: http://localhost:8001/metrics

## License

[License information]