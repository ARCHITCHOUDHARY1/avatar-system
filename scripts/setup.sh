#!/bin/bash

#############################################
# Avatar System - Complete Setup Script
# Assignment Requirement: Reproducible Setup
#############################################

echo "======================================"
echo "Avatar System Orchestrator - Setup"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo -e "\n${YELLOW}[1/7] Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✅ Python $python_version (>= 3.10)${NC}"
else
    echo -e "${RED}❌ Python >= 3.10 required. Found: $python_version${NC}"
    exit 1
fi

# Step 2: Check GPU availability
echo -e "\n${YELLOW}[2/7] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo -e "${GREEN}✅ GPU detected: $gpu_name${NC}"
    export DEVICE=cuda
else
    echo -e "${YELLOW}⚠️  No GPU detected. Using CPU (slower)${NC}"
    export DEVICE=cpu
fi

# Step 3: Create virtual environment
echo -e "\n${YELLOW}[3/7] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Step 4: Install dependencies
echo -e "\n${YELLOW}[4/7] Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 5: Create .env file
echo -e "\n${YELLOW}[5/7] Configuring environment...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    
    # Auto-configure based on detected hardware
    if [ "$DEVICE" == "cuda" ]; then
        sed -i "s/DEVICE=.*/DEVICE=cuda/" .env
        sed -i "s/USE_FP16=.*/USE_FP16=true/" .env
    else
        sed -i "s/DEVICE=.*/DEVICE=cpu/" .env
        sed -i "s/USE_FP16=.*/USE_FP16=false/" .env
    fi
    
    echo -e "${GREEN}✅ .env configured for $DEVICE${NC}"
else
    echo -e "${GREEN}✅ .env already exists${NC}"
fi

# Step 6: Download models
echo -e "\n${YELLOW}[6/7] Downloading models...${NC}"
python scripts/download_models.py
echo -e "${GREEN}✅ Models downloaded${NC}"

# Step 7: Verify installation
echo -e "\n${YELLOW}[7/7] Verifying installation...${NC}"
python -c "
import torch
import transformers
import langgraph
import gradio

print('  PyTorch:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  CUDA version:', torch.version.cuda)
    print('  GPU:', torch.cuda.get_device_name(0))
print('  Transformers:', transformers.__version__)
print('  Gradio:', gradio.__version__)
"

echo -e "${GREEN}✅ Installation verified${NC}"

# Summary
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run Gradio UI: python main.py --mode gradio"
echo "  3. Or run CLI: python main.py --mode cli --input-audio audio.wav --input-image face.jpg"
echo ""
echo "Device: $DEVICE"
echo "Ready for avatar generation! 🎭"
echo ""
