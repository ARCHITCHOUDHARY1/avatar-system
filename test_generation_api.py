"""
Test avatar generation endpoint with real files
"""
import requests
import json
import time

# Server URL
BASE_URL = "http://localhost:8005"

# File paths
IMAGE_PATH = r"d:\avatar-system-orchestrator\data\inputs\images\test_avatar.jpg"
AUDIO_PATH = r"d:\avatar-system-orchestrator\data\inputs\audio\voice_11-01-2026_16-20-43"

# Generation request
payload = {
    "audio_path": AUDIO_PATH,
    "image_path": IMAGE_PATH,
    "fps": 25,
    "resolution": [512, 512]
}

print("[TEST] Starting avatar generation test...")
print(f"[TEST] Image: {IMAGE_PATH}")
print(f"[TEST] Audio: {AUDIO_PATH}")
print(f"[TEST] Settings: 25 FPS, 512x512 resolution")
print("-" * 60)

try:
    print("\n[TEST] Sending POST request to /api/v1/generate...")
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        json=payload,
        timeout=600  # 10 minutes timeout
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n[TEST] Response Status: {response.status_code}")
    print(f"[TEST] Generation Time: {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n[SUCCESS] Avatar generated successfully!")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[ERROR] Generation failed!")
        print(f"Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("\n[ERROR] Request timed out (>10 minutes)")
except requests.exceptions.ConnectionError:
    print("\n[ERROR] Could not connect to server. Is it running on port 8005?")
except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
