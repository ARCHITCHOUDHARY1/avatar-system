# Deployment Guide: Avatar System Orchestrator

## Hardware Requirements

### Minimum Specifications
- **CPU**: 4+ cores (Intel i5/Ryzen 5 or better)
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA T4 (8GB VRAM) or better
- **Storage**: 20GB SSD

### Recommended Specifications
- **CPU**: 8+ cores (Intel i7/Ryzen 7)
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA A100 (40GB/80GB) or RTX 4090
- **Storage**: 50GB NVMe SSD

---

## Cost Analysis (Cloud Deployment)

### Per-Minute Pricing (AWS)

| GPU Instance | On-Demand | Spot | Processing Time* | Cost/Minute** |
|--------------|-----------|------|------------------|----------------|
| **g4dn.xlarge (T4)** | $0.526/hr | $0.158/hr | ~60-90 sec | **$0.006** |
| **g5.xlarge (A10G)** | $1.006/hr | $0.302/hr | ~30-45 sec | **$0.008** |
| **p4d.24xlarge (A100)** | $32.77/hr | ~$10/hr | ~15-20 sec | **$0.055** |

\* For 512x512 @ 25 FPS with 10-second audio  
\*\* Based on actual processing time + overhead

### Monthly Cost Estimates

**1,000 videos/month @ 512x512:**
- T4 (Spot): ~$6-10/month
- A10G (Spot): ~$8-12/month
- A100 (On-Demand): ~$55-80/month

---

## Scaling Strategy (1–100 Concurrent Users)

### Architecture Overview

```
                    ┌─────────────┐
                    │   ALB/NLB   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼────┐      ┌────▼────┐     ┌────▼────┐
     │  API    │      │  API    │     │  API    │
     │ Server  │      │ Server  │     │ Server  │
     │ (CPU)   │      │ (CPU)   │     │ (CPU)   │
     └────┬────┘      └────┬────┘     └────┬────┘
          │                │                │
     ┌────▼──────────────────────────────────┐
     │         Redis Queue (Celery)          │
     └────┬──────────────┬───────────────────┘
          │              │              
     ┌────▼────┐    ┌────▼────┐    
     │ Worker  │    │ Worker  │    ... (Auto-scale)
     │ + GPU   │    │ + GPU   │    
     └─────────┘    └─────────┘    
```

### Implementation Strategy

#### 1-10 Users: Single GPU Instance
- **Setup**: 1 FastAPI server + 1 T4 worker
- **Cost**: ~$0.16/hr (spot)
- **Throughput**: ~40-60 videos/hour

#### 10-50 Users: Queue-Based System
- **Setup**: 
  - 2-3 API servers (t3.medium, CPU-only)
  - Redis Queue (ElastiCache t3.micro)
  - 3-5 GPU workers (g4dn.xlarge pool)
- **Auto-scaling**: Based on queue depth
  - If queue > 10 → Spin up worker
  - If queue < 2 → Terminate worker
- **Cost**: ~$1-3/hr active time
- **Throughput**: ~150-250 videos/hour

#### 50-100 Users: Kubernetes Cluster
- **Setup**:
  - EKS cluster with KEDA autoscaler
  - 5-10 API pods (CPU)
  - 10-20 GPU worker pods (auto-scale)
  - Horizontal Pod Autoscaler (HPA)
- **Metrics**: 
  - Target: Queue depth < 5
  - Scale up: >10 jobs pending
  - Scale down: <3 jobs pending (5 min cooldown)
- **Cost**: ~$5-15/hr peak time
- **Throughput**: ~500+ videos/hour

---

## Deployment Steps

### Option 1: Docker Deployment

```bash
# Build image
docker build -t avatar-system:latest .

# Run with GPU
docker run --gpus all -p 8005:8005 \
  -v $(pwd)/models:/app/models \
  -e MISTRAL_API_KEY=your_key \
  avatar-system:latest
```

### Option 2: Kubernetes (with GPU support)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: avatar-worker
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: worker
        image: avatar-system:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Option 3: AWS EC2 Auto Scaling

1. Create AMI with pre-installed models
2. Launch Template with GPU instance type
3. Auto Scaling Group with scaling policies:
   - Scale out: Queue depth > 10
   - Scale in: Queue depth < 2

---

## Performance Optimization Checklist

- [x] **FP16 Optimization**: 2x speedup
- [x] **TensorRT Compilation**: 4-6x speedup (GPU only)
- [ ] **INT8 Quantization**: 2-4x additional speedup
- [ ] **Model Distillation**: 50% size reduction
- [ ] **Batch Processing**: Process multiple frames together
- [ ] **Warm Pool**: Keep models loaded in memory

---

## Monitoring & Alerts

### Key Metrics to Track

1. **Latency**
   - Target: <60s for 512x512 (T4)
   - Alert if >90s

2. **FPS**
   - Target: ≥30 FPS
   - Alert if <25 FPS

3. **Queue Depth**
   - Normal: <10
   - Alert if >50

4. **GPU Memory**
   - Normal: <7GB (T4)
   - Alert if >7.5GB

5. **Error Rate**
   - Target: <1%
   - Alert if >5%

### Recommended Tools
- **Metrics**: Prometheus + Grafana
- **Logging**: CloudWatch or ELK Stack
- **Tracing**: LangFuse (already integrated)
- **Alerts**: PagerDuty or Opsgenie

---

## Security Considerations

1. **API Authentication**: Implement JWT or API keys
2. **Rate Limiting**: 10 req/min per user
3. **Input Validation**: Max file sizes (10MB image, 50MB audio)
4. **Output Storage**: S3 with signed URLs (24hr expiry)
5. **Network**: VPC with security groups

---

## Backup & Recovery

- **Models**: Daily backup to S3 (versioned)
- **Database**: Automated snapshots (RDS)
- **User Uploads**: S3 lifecycle policy (delete after 7 days)
- **Disaster Recovery**: Multi-region replication

---

## Environment Variables

```bash
# Core
MISTRAL_API_KEY=your_key_here

# Performance
ENABLE_TENSORRT=true
ENABLE_FP16=true
MAX_BATCH_SIZE=4

# Infrastructure
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@host/db

# Monitoring
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
```

---

## Troubleshooting

**Issue**: Out of GPU memory  
**Solution**: Reduce resolution to 256x256 or enable INT8 quantization

**Issue**: Slow performance (<15 FPS)  
**Solution**: Enable TensorRT and FP16, verify GPU is being used

**Issue**: Queue backing up  
**Solution**: Add more GPU workers or reduce concurrent requests

---

## Support & Resources

- **Documentation**: See `ARCHITECTURE.md` and `README.md`
- **API Reference**: http://localhost:8005/docs
- **GitHub Issues**: [Project Repository]
- **Technical Analysis**: See `technical_analysis.md`
