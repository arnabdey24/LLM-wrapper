#!/bin/bash

# Gemma API Service Startup Script

echo "🚀 Starting Gemma API Service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Check for .env file
if [ ! -f .env ]; then
    echo "📝 No .env file found. Using default model (microsoft/DialoGPT-medium)"
    echo "   To use Gemma or other models, copy .env.example to .env and configure"
fi

echo "📦 Building and starting containers..."
docker-compose up --build -d

echo "⏳ Waiting for service to be ready..."
sleep 10

# Check if service is healthy
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Service is ready!"
        echo ""
        echo "🌐 API Documentation: http://localhost:8000/docs"
        echo "🏥 Health Check: http://localhost:8000/health"
        echo "📊 Logs: docker-compose logs -f gemma-api"
        echo ""
        echo "🧪 Run tests: python test_api.py"
        break
    fi
    echo "Waiting for service... ($i/30)"
    sleep 2
done

if [ $i -eq 30 ]; then
    echo "❌ Service failed to start. Check logs with: docker-compose logs gemma-api"
    exit 1
fi