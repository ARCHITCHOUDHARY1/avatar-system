# Avatar System Orchestrator - Complete Guide

## **📊 Project Overview**

This is an **open-source avatar generation system** that creates realistic talking avatars from static images and audio using AI models orchestrated through LangGraph.

### **Key Features**
- 🎭 **LangGraph-based** stateful workflow orchestration  
- 🧠 **Mistral-7B** for dynamic avatar control parameters
- 🎬 **Hybrid model selection**: Wav2Lip (CPU) or SadTalker (GPU)
- 🔊 **Multi-modal emotion detection** (audio + facial)
- 🌐 **REST API + WebSocket** for real-time streaming
- 📊 **Comprehensive observability** with Langfuse integration

---

## **🚀 Quick Start**

### Prerequisites
- Python 3.9+
- CUDA 12.1+ (optional, for GPU acceleration)
- FFmpeg installed
- 8GB+ RAM (16GB recommended)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd avatar-system-orchestrator

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Download models
python scripts/download_models.py
```

### Run Server
```bash
python main.py
# Server starts on http://localhost:8005
```

### Test Generation
```bash
# Place test files in data/inputs/
python test_generation_api.py
```

---

## **🏗️ System Architecture**

### LangGraph Pipeline

```
START → Audio Processing → Emotion Detection → Mistral Controller 
      → Video Generation → Quality Enhancement → END
```

**Each node:**
- Receives shared `AvatarState`
- Performs specific processing
- Updates state with results
- Passes to next node

### Hybrid Model System

```python
# Automatic selection based on hardware:
if GPU available:
    use SadTalker  # High quality, GPU-accelerated
else:
    use Wav2Lip    # CPU-optimized, faster
```

---

## **📁 Project Structure** 

```
avatar-system-orchestrator/
├── configs/           # YAML configurations
├── data/              # Input/output files
│   ├── inputs/        # Audio, images, videos
│   └── outputs/       # Generated videos
├── models/            # AI model implementations
│   ├── wav2lip_model.py
│   └── sadtalker_integration.py
├── src/
│   ├── orchestrator/  # LangGraph pipeline (CORE)
│   │   ├── graph_builder.py
│   │   └── workflow_nodes.py
│   ├── api/           # FastAPI REST endpoints
│   ├── video_processor/  # Video generation
│   └── audio_processor/  # Audio processing
├── web/               # Web UI
│   ├── index.html
│   └── static/
├── main.py            # Entry point
└── requirements.txt
```

---

## **🎯 Core Components**

### 1. **Workflow Nodes** (`src/orchestrator/workflow_nodes.py`)

#### Audio Processing Node
- Transcribes audio with Whisper
- Extracts features (energy, spectral)
- Detects voice activity

#### Emotion Detection Node
- Analyzes audio emotion
- Detects facial emotion (if image provided)
- Combines signals into primary emotion

#### Mistral Controller Node ⭐
- Uses Mistral-7B LLM to generate avatar controls
- Maps emotion + audio → facial parameters
- Output: `blink_rate`, `head_tilt`, `expression_intensity`, etc.

#### Video Generation Node
- Applies controls to selected model
- Uses Wav2Lip (CPU) or SadTalker (GPU)
- Generates synchronized talking head video

#### Quality Enhancement Node
- Enhances faces with GFPGAN
- Reduces flicker
- Improves visual quality

### 2. **Hybrid Model Selection** (`src/video_processor/hybrid_generator.py`)

```python
class HybridVideoGenerator:
    def generate(self, image, audio, ...):
        # Auto-detect hardware
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            return self._generate_sadtalker(...)
        else:
            return self._generate_wav2lip(...)
```

**Benefits:**
- Automatic model selection
- Graceful fallback (SadTalker → Wav2Lip)
- Optimized for available hardware

### 3. **State Management** (`src/orchestrator/graph_builder.py`)

```python
class AvatarState(TypedDict):
    # Inputs
    audio_input: str
    image_input: Any
    output_path: str
    
    # Processing
    transcribed_text: str
    audio_features: Dict
    emotion: str
    confidence: float
    avatar_control: Dict  # From Mistral
    
    # Outputs
    final_video: str
    enhanced_video: str
    
    # Metadata
    errors: List[str]
    performance: Dict[str, float]
```

---

## **🔌 API Reference**

### REST Endpoints

#### **Generate Avatar**
```http
POST /api/v1/generate
Content-Type: application/json

{
  "image_path": "data/inputs/images/face.jpg",
  "audio_path": "data/inputs/audio/voice.wav",
  "fps": 25,
  "resolution": [512, 512]
}

Response:
{
  "video_path": "/outputs/videos/avatar_123.mp4",
  "duration": 30.5,
  "status": "completed"
}
```

#### **Upload Files**
```http
POST /api/v1/upload/image
POST /api/v1/upload/audio  
POST /api/v1/upload/video

Content-Type: multipart/form-data
```

#### **System Status**
```http
GET /api/v1/status
GET /api/v1/stats/dashboard
GET /api/v1/stats/performance
```

### WebSocket Streaming
```javascript
ws = new WebSocket('ws://localhost:8005/ws');
ws.send(JSON.stringify({
  type: 'start_stream',
  image: '...',
  audio_stream: true
}));
```

---

## **🧪 Testing**

### End-to-End Test
```bash
# Place test files:
# - data/inputs/images/test_avatar.jpg
# - data/inputs/audio/voice_sample.wav

python test_generation_api.py
```

### Unit Tests
```bash
pytest tests/
```

---

## **⚙️ Configuration**

### Environment Variables (`.env`)

```env
# API Keys
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key

# Model Settings
DEFAULT_FPS=25
DEFAULT_RESOLUTION=512

# Observability (optional)
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
models:
  wav2lip:
    checkpoint: models/wav2lip/checkpoints/wav2lip_gan.pth
    device: cpu
  
  sadtalker:
    checkpoint: models/sadtalker/checkpoints/
    device: cuda
    
  mistral:
    model: mistral-7b-instruct
    temperature: 0.7
```

---

## **📊 Observability (Langfuse)**

### Setup
1. Sign up at https://cloud.langfuse.com
2. Create project and get API keys
3. Add keys to `.env`
4. All pipeline runs auto-traced!

### Features
- 🔍 Complete execution tracing
- 📊 Performance monitoring
- 🐛 Error debugging with context
- 🤖 LLM prompt/response logging

### View Traces
```
Dashboard → Traces → Select execution → View details
```

---

## **💡 Key Concepts**

### Mistral Controller
**Why Mistral-7B?**
- Generates **intelligent, context-aware** avatar controls
- Maps complex emotion + audio patterns → precise parameters
- More natural than rule-based systems

**Example Output:**
```json
{
  "blink_rate": 0.8,       // 0.8 blinks/sec (energetic)
  "head_tilt": 0.3,        // Slight upward tilt (positive)
  "expression_intensity": 0.9,  // Strong expression
  "mouth_openness": 0.7,   // Wide mouth (loud audio)
  "eyebrow_raise": 0.4     // Moderate surprise
}
```

### Hybrid Model Selection
- **CPU systems**: Automatic Wav2Lip (optimized, faster)
- **GPU systems**: Automatic SadTalker (higher quality)
- **Fallback**: SadTalker fails → Wav2Lip

--- ## **🔧 Troubleshooting**

### Common Issues

**1. Model Download Fails**
```bash
# Manual download:
python scripts/download_models.py
```

**2. CUDA Out of Memory**
```python
# Reduce resolution in config:
DEFAULT_RESOLUTION=256  # Instead of 512
```

**3. FFmpeg Not Found**
```bash
# Install FFmpeg:
# Windows: choco install ffmpeg
# Linux: apt-get install ffmpeg
# Mac: brew install ffmpeg
```

**4. Generation Timeout**
- CPU generation takes 3-5 minutes for 10-second video
- Increase timeout or use GPU for faster processing

---

## **📈 Performance**

### Current Benchmarks (Local CPU)
- **Latency**: 3-5 minutes per 10-second video
- **FPS**: ~0.1 FPS equivalent (not real-time)
- **Model**: Wav2Lip (CPU-optimized)

### Optimization Roadmap
- [ ] TensorRT optimization (10-100x speedup)
- [ ] FP16 quantization (2x faster)
- [ ] Batching for concurrent requests
- [ ] Redis caching layer
- [ ] Target: 30 FPS real-time

---

## **🎓 Learning Resources**

### Official Docs
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Mistral AI API](https://docs.mistral.ai/)
- [Wav2Lip Paper](https://arxiv.org/abs/2008.10010)
- [SadTalker Paper](https://arxiv.org/abs/2211.12194)

### This Project
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture
- [API.md](docs/API.md) - Complete API reference
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide

---

## **🤝 Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## **📝 License**

This project uses open-source models:
- **Wav2Lip**: Apache 2.0
- **SadTalker**: MIT
- **Mistral-7B**: Apache 2.0
- **Whisper**: MIT
- **PyTorch**: BSD-3-Clause

See individual model licenses for details.

---

## **🆘 Support**

- **Issues**: https://github.com/your-repo/issues
- **Discussions**: https://github.com/your-repo/discussions
- **Email**: support@your-domain.com

---

## **🎯 Roadmap**

### Phase 1: Current (✅ Complete)
- LangGraph pipeline orchestration
- Hybrid Wav2Lip/SadTalker support
- Mistral controller integration
- REST API + WebUI
- Langfuse observability

### Phase 2: Performance (In Progress)
- TensorRT optimization
- FP16 quantization
- Real-time processing (30 FPS)
- Quality metrics (LPIPS, SSIM)

### Phase 3: Production (Planned)
- Kubernetes deployment
- Auto-scaling
- Monitoring stack (Prometheus/Grafana)
- Cost optimization
- Multi-GPU support

### Phase 4: Advanced Features (Future)
- 3D avatar support
- Real-time video input
- Multi-language support
- Custom voice cloning
- Emotion transfer

---

**Built with ❤️ using LangGraph + Mistral + PyTorch**
