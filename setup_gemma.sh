#!/bin/bash

# Setup script for Gemma model

echo "🔧 Gemma Model Setup"
echo "===================="
echo ""

echo "To use Gemma models, you need:"
echo "1. Request access at: https://huggingface.co/google/gemma-2b-it"
echo "2. Get a token from: https://huggingface.co/settings/tokens"
echo ""

read -p "Do you have access to Gemma and a Hugging Face token? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Enter your Hugging Face token: " HF_TOKEN
    
    if [ -z "$HF_TOKEN" ]; then
        echo "❌ No token provided. Exiting."
        exit 1
    fi
    
    # Create .env file
    cat > .env << EOF
# Model Configuration
MODEL_NAME=google/gemma-2b-it
HUGGINGFACE_TOKEN=$HF_TOKEN
EOF
    
    echo "✅ .env file created with Gemma configuration"
    echo ""
    echo "You can now run: ./start.sh"
    
else
    echo ""
    echo "No problem! The service will use microsoft/DialoGPT-medium by default."
    echo "This model works great and doesn't require any authentication."
    echo ""
    echo "You can run: ./start.sh"
fi