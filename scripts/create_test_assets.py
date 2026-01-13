
import os
import cv2
import numpy as np
import wave
import struct

def create_dummy_image():
    # Create a 512x512 black image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a white circle
    cv2.circle(img, (256, 256), 100, (255, 255, 255), -1)
    
    path = "d:/avatar-system-orchestrator/data/inputs/images/test_avatar.jpg"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Created image: {path}")
    return path

def create_dummy_audio():
    # Create 1 second of silence
    duration = 1.0 
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    path = "d:/avatar-system-orchestrator/data/inputs/audio/test_audio.wav"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1) # Mono
        wav_file.setsampwidth(2) # 2 bytes
        wav_file.setframerate(sample_rate)
        
        # Write silence
        data = struct.pack('<' + ('h'*num_samples), *([0]*num_samples))
        wav_file.writeframes(data)
        
    print(f"Created audio: {path}")
    return path

if __name__ == "__main__":
    create_dummy_image()
    create_dummy_audio()
