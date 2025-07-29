# Gemma 3 FastAPI Service

A FastAPI service that wraps the Gemma 3 model with support for both text and image inputs, optimized for MacBook Pro M4.

## Features

- **Text Generation**: Generate text from text prompts
- **Image + Text**: Process both image and text inputs together
- **M4 Optimization**: Leverages Metal Performance Shaders (MPS) for optimal performance on M4 MacBooks
- **Docker Support**: Easy deployment with Docker Compose
- **Health Checks**: Built-in health monitoring
- **CORS Support**: Ready for web applications

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up --build

# For production with nginx reverse proxy
docker-compose --profile production up --build
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Text Generation
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain quantum computing in simple terms",
    "max_length": 256,
    "temperature": 0.7
  }'
```

### Text + Image Generation
```bash
curl -X POST "http://localhost:8000/generate-with-image" \
  -F "text=Describe what you see in this image" \
  -F "image=@your_image.jpg" \
  -F "max_length=256"
```

## Configuration

### Environment Variables
- `MODEL_NAME`: Model to use (default: microsoft/DialoGPT-medium)
- `HUGGINGFACE_TOKEN`: Your HF token for gated models (optional)
- `TRANSFORMERS_CACHE`: Cache directory for models
- `HF_HOME`: Hugging Face cache directory

### Model Configuration

The service supports multiple models via environment variables:

**Default Model**: `microsoft/DialoGPT-medium` (no authentication required)

**Available Models**:
- `microsoft/DialoGPT-medium` - Good conversational model (default)
- `microsoft/phi-2` - Small but capable model  
- `distilgpt2` - Lightweight GPT-2 variant
- `google/gemma-2b-it` - Requires Hugging Face token and access request

**Using Gemma Models**:
1. Request access at https://huggingface.co/google/gemma-2b-it
2. Get a token from https://huggingface.co/settings/tokens
3. Copy `.env.example` to `.env` and set your token:
```bash
cp .env.example .env
# Edit .env with your MODEL_NAME and HUGGINGFACE_TOKEN
```

## Performance Optimization for M4 MacBook Pro

- **MPS Support**: Automatically detects and uses Metal Performance Shaders
- **Memory Efficient**: Uses float16 precision when possible
- **Model Caching**: Persistent model cache to avoid re-downloading
- **Resource Limits**: Configured memory limits in Docker Compose

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use a smaller model
2. **Slow Performance**: Ensure MPS is available and being used
3. **Model Download**: First run will download the model (~5GB)

### Logs
```bash
# View logs
docker-compose logs -f gemma-api
```

## Development

### Adding New Models
Modify the `model_name` variable in `main.py` and update requirements if needed.

### Custom Endpoints
Add new endpoints in `main.py` following the existing pattern.

## License

MIT License