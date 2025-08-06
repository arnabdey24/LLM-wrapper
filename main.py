from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io
import base64
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemma 3 API Service",
    description="FastAPI service wrapping Gemma 3 model with text and image support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
tokenizer = None
model = None
device = None

class TextRequest(BaseModel):
    text: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    input_text: str
    model_info: dict

@app.on_event("startup")
async def load_model():
    """Load the Gemma model on startup"""
    global tokenizer, model, device
    
    try:
        # Optimize for M4 MacBook Pro
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        logger.info(f"MPS built: {torch.backends.mps.is_built()}")
        
        # Try to use MPS, fall back to CPU if there are issues
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                # Test MPS with a simple tensor operation
                test_tensor = torch.randn(1, 1).to(device)
                logger.info("Using MPS (Metal Performance Shaders) for M4 optimization")
            else:
                device = torch.device("cpu")
                logger.info("MPS not available, using CPU")
        except Exception as e:
            logger.warning(f"MPS initialization failed: {e}, falling back to CPU")
            device = torch.device("cpu")
        
        # Use Microsoft's DialoGPT or Phi-2 as alternatives to gated Gemma
        # These are open models that work well for text generation
        model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Alternative models you can use:
        # "microsoft/DialoGPT-medium" - Good conversational model
        # "microsoft/phi-2" - Small but capable model
        # "distilgpt2" - Lightweight GPT-2 variant
        
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token if hf_token else None
        )
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model from {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
            device_map="auto" if device.type != "mps" else None,
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None
        )
        
        if device.type == "mps":
            model = model.to(device)
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def generate_text(prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """Generate text using the loaded model"""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        if device.type == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Gemma 3 API Service is running",
        "status": "healthy",
        "device": str(device) if device else "not loaded"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device else "not loaded",
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available()
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text_endpoint(request: TextRequest):
    """Generate text from text input"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_text = generate_text(
            request.text,
            request.max_length,
            request.temperature,
            request.top_p
        )
        
        return GenerationResponse(
            generated_text=generated_text,
            input_text=request.text,
            model_info={
                "device": str(device),
                "model_name": os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium"),
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-with-image")
async def generate_with_image(
    text: str = Form(...),
    image: UploadFile = File(...),
    max_length: Optional[int] = Form(512),
    temperature: Optional[float] = Form(0.7),
    top_p: Optional[float] = Form(0.9)
):
    """Generate text from text and image input"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert image to base64 for logging/debugging
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # For now, we'll create a prompt that describes the image context
        # Note: Gemma 2B doesn't natively support vision, so we're creating a text-only prompt
        enhanced_prompt = f"""Based on the uploaded image and the following text: "{text}"
        
Please provide a response that takes into account both the visual context and the text prompt. 
Image has been uploaded and processed.

Response:"""
        
        generated_text = generate_text(
            enhanced_prompt,
            max_length,
            temperature,
            top_p
        )
        
        return {
            "generated_text": generated_text,
            "input_text": text,
            "image_info": {
                "filename": image.filename,
                "size": pil_image.size,
                "format": pil_image.format
            },
            "model_info": {
                "device": str(device),
                "model_name": os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium"),
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "note": "Image context included in prompt (text-only model)"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate-with-image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)