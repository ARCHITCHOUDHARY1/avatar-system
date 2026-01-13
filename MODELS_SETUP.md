# Model Setup Guide

## 📦 **Model Files Not Included in Git**

Due to GitHub's file size limits, **large model files are excluded** from this repository. You need to download them separately.

**Required Models:**
- Wav2Lip GAN model (~415 MB)
- SadTalker checkpoints (optional, for GPU)
- GFPGAN weights (optional, for quality enhancement)

---

## 🚀 **Quick Setup**

### Option 1: Automated Download (Recommended)

```bash
# Download all required models
python scripts/download_models.py
```

This script will:
- ✅ Download Wav2Lip model to `models/wav2lip/checkpoints/`
- ✅ Download SadTalker (if GPU available)
- ✅ Download GFPGAN weights
- ✅ Verify checksums

---

### Option 2: Manual Download

#### Wav2Lip Model (Required for CPU)

```bash
# Create directory
mkdir -p models/wav2lip/checkpoints

# Download model (415 MB)
wget https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth \
  -O models/wav2lip/checkpoints/wav2lip_gan.pth
```

**Or use PowerShell:**
```powershell
New-Item -ItemType Directory -Force -Path models\wav2lip\checkpoints
Invoke-WebRequest -Uri "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" `
  -OutFile "models\wav2lip\checkpoints\wav2lip_gan.pth"
```

---

#### SadTalker Models (Optional, for GPU)

```bash
# Coming soon - GPU optimization phase
# For now, system uses Wav2Lip on CPU
```

---

## ✅ **Verify Installation**

After downloading models:

```bash
# Check if models exist
python -c "from pathlib import Path; print('✅ Wav2Lip ready!' if Path('models/wav2lip/checkpoints/wav2lip_gan.pth').exists() else '❌ Download Wav2Lip model')"

# Or check manually
ls models/wav2lip/checkpoints/wav2lip_gan.pth  # Should show 415 MB file
```

---

## 📝 **What's Excluded from Git (.gitignore)**

The following large files are **NOT** in the repository:

```
# Model checkpoints
models/wav2lip/checkpoints/*.pth
models/sadtalker/**/*.pth
gfpgan/weights/*.pth

# Data files
data/inputs/**/*
data/outputs/**/*

# Caches
models/cache/
```

---

## 🔧 **Troubleshooting**

### Download Fails

**Try alternative mirror:**
```bash
# Hugging Face mirror (if available)
wget https://huggingface.co/spaces/Wav2Lip/Wav2Lip/resolve/main/wav2lip_gan.pth
```

### Python Download Script

If you need to manually download using Python:

```python
import urllib.request
from pathlib import Path

url = "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
output = Path("models/wav2lip/checkpoints/wav2lip_gan.pth")
output.parent.mkdir(parents=True, exist_ok=True)

print("Downloading Wav2Lip model (415 MB)...")
urllib.request.urlretrieve(url, output)
print("✅ Download complete!")
```

---

## 📊 **Model Requirements**

| Model | Size | Required | Used For |
|-------|------|----------|----------|
| Wav2Lip GAN | 415 MB | ✅ Yes | CPU video generation |
| SadTalker | ~2 GB | ❌ Optional | GPU high-quality (future) |
| GFPGAN | ~350 MB | ❌ Optional | Quality enhancement |

---

## 🎯 **After Setup**

Once models are downloaded:

```bash
# Test the system
python test_generation_api.py

# Start the server
python main.py
```

---

## 📚 **More Information**

- **Wav2Lip Paper**: https://arxiv.org/abs/2008.10010
- **Model Weights**: See download script for URLs
- **System Requirements**: See README.md

---

**Need help?** Open an issue: https://github.com/ARCHITCHOUDHARY1/avatar-system/issues
