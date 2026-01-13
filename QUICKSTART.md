# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Clone & Install (2 min)
```bash
git clone <repository-url>
cd avatar-system-orchestrator
pip install -r requirements.txt
pip install gfpgan basicsr facexlib kornia yacs face-alignment
```

### Step 2: Setup SadTalker (2 min)
```bash
cd models
git clone https://github.com/OpenTalker/SadTalker.git sadtalker_source
cd ..
python setup_sadtalker.py
python download_sadtalker_checkpoints.py
```

### Step 3: Generate! (1 min)
```bash
# Test with sample
python test_sadtalker.py

# Or use your own files
python main.py --mode cli \
  --input-audio your_voice.wav \
  --input-image your_face.jpg \
  --output avatar.mp4
```

---

## 📋 Command Cheat Sheet

### Generate Avatar
```bash
# CLI
python main.py --mode cli --input-audio voice.wav --input-image face.jpg --output avatar.mp4

# API Server
python main.py --mode api  # Access at http://localhost:8005
```

### Test & Debug
```bash
python test_sadtalker.py              # Test SadTalker
pytest tests/                         # Run tests
python -c "import torch; print(torch.cuda.is_available())"  # Check GPU
```

### Common Fixes
```bash
pip install kornia                    # Fix kornia error
pip install gfpgan                    # Fix gfpgan error
pip install yacs face-alignment       # Fix SadTalker dependencies
```

---

## 📁 Input Requirements

### Audio
- **Format**: WAV, MP3
- **Duration**: Any (longer = slower processing)
- **Quality**: Clear speech recommended

### Image
- **Format**: JPG, PNG
- **Resolution**: 256x256 to 1024x1024
- **Content**: Clear, front-facing face
- **Examples**: `models/sadtalker_source/examples/source_image/`

---

## ⚙️ Configuration

### Change Port
Edit `main.py` or use environment variable:
```bash
export PORT=8080
python main.py --mode api
```

### GPU/CPU
Automatically detected. Force CPU:
```python
import torch
torch.device('cpu')
```

### Output Quality
Edit `configs/inference.yaml`:
```yaml
resolution: 512  # 256 or 512
fps: 25         # frames per second
enhance: true   # GFPGAN enhancement
```

---

## 🎯 Expected Output

```
models/sadtalker/outputs/
└── test_output.mp4         # Generated video
    └── tmp/                # Intermediate files
        ├── first_frame_dir/
        └── coefficients/
```

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `kornia not found` | `pip install kornia` |
| `Face detection fails` | Use example images |
| `Very slow` | Normal on CPU; use GPU |
| `Out of memory` | Reduce resolution to 256 |
| `Port already in use` | Change port in main.py |

**Full guide**: [SETUP.md](SETUP.md)

---

## 📞 Need Help?

1. Check [SETUP.md](SETUP.md) for detailed troubleshooting
2. Run `python test_sadtalker.py` to verify installation
3. See logs for error details
4. Open an issue with logs

---

**Total Setup Time**: ~10 minutes (including downloads)
