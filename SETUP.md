# Complete Setup Guide - Avatar System Orchestrator

This guide will walk you through setting up the Avatar System Orchestrator from scratch, including all dependencies and the SadTalker integration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [SadTalker Setup](#sadtalker-setup)
- [Running the System](#running-the-system)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.11 (recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 15GB free space for models
- **GPU**: Optional (CUDA 11.8+ for GPU acceleration)

### Required Software

1. **Python 3.11**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure `pip` is installed

2. **FFmpeg**
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`

3. **Git**
   - Download from [git-scm.com](https://git-scm.com/downloads)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd avatar-system-orchestrator
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (API server)
- LangGraph & LangChain (orchestration)
- Transformers & Torch (ML models)
- Audio processing libraries (librosa, soundfile)
- And all other core dependencies

### Step 4: Install SadTalker Dependencies

The following packages are required for SadTalker but may not be in `requirements.txt`:

```bash
pip install gfpgan basicsr facexlib kornia yacs face-alignment joblib scikit-image av resampy tb-nightly
```

---

## SadTalker Setup

SadTalker is the core video generation engine. Follow these steps carefully:

### Step 1: Clone SadTalker Repository

```bash
cd models
git clone https://github.com/OpenTalker/SadTalker.git sadtalker_source
cd ..
```

### Step 2: Install SadTalker Source Code

Run the setup script to copy SadTalker source files to the correct location:

```bash
python setup_sadtalker.py
```

This script:
- Copies `sadtalker_source/src` to `models/sadtalker/src`
- Creates necessary `__init__.py` files
- Sets up the module structure

### Step 3: Download SadTalker Model Checkpoints

```bash
python download_sadtalker_checkpoints.py
```

This downloads (~880MB):
- `SadTalker_V0.0.2_256.safetensors` (725MB)
- `mapping_00109-model.pth.tar` (156MB)

Files are saved to `models/sadtalker/checkpoints/`

### Step 4: Verify Configuration Files

Check that config files exist in `models/sadtalker/src/config/`:
- `auido2pose.yaml` (note the typo in filename - this is from SadTalker)
- `auido2exp.yaml`
- `facerender.yaml`
- `facerender_still.yaml`

These should be automatically present after Step 2.

---

## Runningthe System

### Quick Test

Verify the installation with a simple test:

```bash
python test_sadtalker.py
```

This will:
1. Import and initialize SadTalker
2. Load all models
3. Generate a test video from sample input

**Expected output**: `models/sadtalker/outputs/test_output.mp4`

### CLI Mode - Generate Avatar

Generate an avatar from your own audio and image:

```bash
python main.py --mode cli \
  --input-audio path/to/your/audio.wav \
  --input-image path/to/your/face.jpg \
  --output output_avatar.mp4
```

**Example:**
```bash
python main.py --mode cli \
  --input-audio voice_input.wav \
  --input-image models/sadtalker_source/examples/source_image/full_body_1.png \
  --output my_avatar.mp4
```

### API Mode - REST Server

Start the FastAPI server:

```bash
python main.py --mode api
```

Access the API:
- **Base URL**: http://localhost:8005
- **API Docs**: http://localhost:8005/docs
- **Frontend**: http://localhost:8005/

### Web Interface

The web interface is available at http://localhost:8005 when running in API mode.

Features:
- Drag-and-drop file upload
- Audio recording
- Real-time generation status
- Download generated videos

---

## Complete Workflow

### 1. Audio Processing
- Loads audio file
- Extracts audio features (energy, duration)
- Skips heavy models (Whisper, WavLM) for speed
- Outputs: `transcribed_text`, `audio_features`

### 2. Emotion Detection
- Analyzes audio with HuBERT
- Optional: Analyzes face image with EMOCA
- Combines audio + face emotions
- Outputs: `emotion`, `confidence`

### 3. Mistral Controller (Optional)
- Uses Mistral AI for dynamic avatar control
- Generates expression parameters based on emotion
- Outputs: `avatar_control` parameters

### 4. Video Generation (SadTalker)
- **Preprocessing**: Detects face, extracts 3D landmarks
- **Audio2Coeff**: Converts audio to 3D facial coefficients
- **Rendering**: Generates video frames with lip-sync
- Outputs: `final_video`

### 5. Quality Enhancement (Optional)
- GFPGAN face enhancement
- Background upsampling
- Outputs: `enhanced_video`

---

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'kornia'`

**Solution:**
```bash
pip install kornia
```

#### 2. `No module named 'gfpgan'`

**Solution:**
```bash
pip install gfpgan basicsr facexlib
```

#### 3. `torchvision.transforms.functional_tensor not found`

This is automatically patched in `models/sadtalker_integration.py`. If you still see this error, ensure you're using the latest code.

#### 4. `RuntimeError: Expected input to have 73 channels, but got 70`

This is fixed by updating `facerender.yaml` and `generate_facerender_batch.py` to use 73 channels. The fix is already applied in the repository.

#### 5. Face Detection Fails

**Error**: `can not detect the landmark from source image`

**Solution**: Use a clear, front-facing image with a visible face. Try example images from:
```
models/sadtalker_source/examples/source_image/
```

#### 6. `numpy.float` AttributeError

This is automatically patched for NumPy 1.20+. The fix is in `models/sadtalker_integration.py`.

#### 7. Slow Performance on CPU

**Expected**: 5-10 minutes per video on CPU
**GPU Acceleration**: Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Getting Help

If you encounter issues:

1. Check the logs for detailed error messages
2. Verify all dependencies are installed: `pip list`
3. Ensure SadTalker checkpoints are downloaded
4. Try the test script: `python test_sadtalker.py`
5. Check existing issues in the repository

---

## Performance Optimization

### For Faster Generation

1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Reduce Resolution**: Use 256x256 instead of 512x512
3. **Skip Enhancement**: Don't use GFPGAN enhancement
4. **Batch Processing**: Process multiple videos sequentially

### Memory Optimization

- Close other applications
- Use smaller batch sizes
- Clear cache between generations

---

## File Locations

```
avatar-system-orchestrator/
├── models/
│   ├── sadtalker/              # SadTalker installation
│   │   ├── src/                # Source code (from setup)
│   │   ├── checkpoints/        # Model weights
│   │   └── outputs/            # Generated videos
│   └── sadtalker_source/       # Cloned repository
├── data/
│   ├── inputs/                 # Your input files
│   └── outputs/                # Generated outputs
├── voice_input.wav             # Sample audio
└── output_avatar.mp4           # Generated video
```

---

## Next Steps

After setup:

1. **Test with Examples**: Use provided sample files
2. **Try Your Own Content**: Upload your audio/images
3. **Explore the API**: Check http://localhost:8005/docs
4. **Monitor Performance**: Use `/api/performance` endpoint
5. **Customize Settings**: Edit config files in `configs/`

---

## Additional Resources

- **SadTalker Docs**: https://github.com/OpenTalker/SadTalker
- **LangGraph Guide**: https://langchain-ai.github.io/langgraph/
- **API Documentation**: http://localhost:8005/docs (when server is running)

---

## Changelog

### Latest Updates (2026-01-11)

- ✅ Fixed SadTalker integration
- ✅ Resolved all dependency conflicts
- ✅ Updated to 73-channel model configuration
- ✅ Added numpy compatibility patches
- ✅ Improved preprocessing pipeline
- ✅ Complete CPU support

---

**Need Help?** Open an issue or check the [documentation](docs/).
