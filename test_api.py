#!/usr/bin/env python3
"""
Simple test script for the Gemma API service
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_text_generation():
    """Test text generation endpoint"""
    print("\nTesting text generation...")
    try:
        payload = {
            "text": "Write a short poem about artificial intelligence",
            "max_length": 200,
            "temperature": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Generated text: {result['generated_text']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Text generation test failed: {e}")
        return False

def test_image_generation():
    """Test image + text generation endpoint"""
    print("\nTesting image + text generation...")
    print("Note: This test requires an image file named 'test_image.jpg'")
    
    try:
        # Create a simple test image if it doesn't exist
        from PIL import Image
        import io
        
        # Create a simple colored rectangle as test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            'image': ('test.jpg', img_bytes, 'image/jpeg')
        }
        data = {
            'text': 'What color is this image?',
            'max_length': 150
        }
        
        response = requests.post(f"{BASE_URL}/generate-with-image", files=files, data=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Generated text: {result['generated_text']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Image generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting API tests...")
    
    # Wait for service to be ready
    print("Waiting for service to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        if test_health():
            break
        time.sleep(1)
        print(f"Waiting... ({i+1}/30)")
    else:
        print("Service not ready after 30 seconds")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Text Generation", test_text_generation),
        ("Image + Text Generation", test_image_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    main()