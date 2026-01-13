from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class GenerationJob(Base):
    __tablename__ = 'generation_jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), unique=True, nullable=False, index=True)
    
    # Input files
    audio_path = Column(String(512))
    image_path = Column(String(512))
    output_path = Column(String(512))
    
    # Status
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Results
    emotion = Column(String(50))
    confidence = Column(Float)
    avatar_controls = Column(JSON)  # Mistral-generated controls
    
    # Performance metrics (Assignment requirement)
    total_time_seconds = Column(Float)
    fps_achieved = Column(Float)
    meets_realtime = Column(Boolean)  # >= 30 FPS
    
    # Stage timings
    audio_processing_time = Column(Float)
    emotion_detection_time = Column(Float)
    mistral_controller_time = Column(Float)
    video_generation_time = Column(Float)
    quality_enhancement_time = Column(Float)
    
    # Quality metrics (Assignment requirement)
    quality_score = Column(Float)  # 0-100
    psnr = Column(Float)
    ssim = Column(Float)
    temporal_consistency = Column(Float)
    lip_sync_error = Column(Float)
    
    # Cost tracking (Assignment: Deployment cost)
    gpu_type = Column(String(20))  # T4, V100, A100, CPU
    compute_cost_usd = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Error tracking
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<GenerationJob(id={self.job_id}, status={self.status}, fps={self.fps_achieved})>"


class PerformanceMetric(Base):
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Hardware info
    device_type = Column(String(10))  # cuda, cpu
    gpu_name = Column(String(100))
    gpu_memory_gb = Column(Float)
    
    # Configuration
    batch_size = Column(Integer)
    resolution = Column(Integer)
    use_fp16 = Column(Boolean)
    use_quantization = Column(Boolean)
    
    # Performance results (Assignment: 30 FPS requirement)
    fps = Column(Float)
    latency_ms = Column(Float)
    passes_realtime = Column(Boolean)  # >= 30 FPS
    
    # Optimizations applied
    optimizations = Column(JSON)  # List of optimizations
    expected_speedup = Column(Float)
    
    # Memory usage
    peak_memory_mb = Column(Float)
    avg_memory_mb = Column(Float)
    
    # Timestamps
    measured_at = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<PerformanceMetric(fps={self.fps}, device={self.device_type})>"


class QualityReport(Base):
    __tablename__ = 'quality_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), index=True)
    
    # Video info
    video_path = Column(String(512))
    fps = Column(Float)
    resolution_width = Column(Integer)
    resolution_height = Column(Integer)
    frame_count = Column(Integer)
    duration_seconds = Column(Float)
    
    # Quality metrics (Assignment requirements)
    overall_score = Column(Float)  # 0-100
    
    # Visual quality
    psnr_mean = Column(Float)
    psnr_std = Column(Float)
    ssim_mean = Column(Float)
    ssim_std = Column(Float)
    
    # Temporal quality (anti-flicker)
    flicker_score = Column(Float)  # Lower = better
    mean_diff = Column(Float)
    temporal_consistency_score = Column(Float)
    
    # Lip sync
    lip_sync_error = Column(Float)  # Lower = better
    audio_visual_correlation = Column(Float)
    
    # Expression richness
    expression_variance = Column(Float)
    expression_range = Column(Float)
    richness_score = Column(Float)
    
    # Artifacts
    blur_mean = Column(Float)
    is_blurry = Column(Boolean)
    alignment_shift = Column(Float)
    is_misaligned = Column(Boolean)
    
    # Full results
    full_metrics = Column(JSON)
    
    # Timestamps
    evaluated_at = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<QualityReport(job_id={self.job_id}, score={self.overall_score})>"


class UserSession(Base):
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), unique=True, nullable=False, index=True)
    
    # User info (if authentication enabled)
    user_id = Column(String(100))
    api_key_hash = Column(String(64))
    
    # Activity
    total_jobs = Column(Integer, default=0)
    successful_jobs = Column(Integer, default=0)
    failed_jobs = Column(Integer, default=0)
    
    # Resource usage
    total_compute_time_seconds = Column(Float, default=0.0)
    total_cost_usd = Column(Float, default=0.0)
    
    # Rate limiting
    requests_last_minute = Column(Integer, default=0)
    requests_last_hour = Column(Integer, default=0)
    last_request_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    last_active_at = Column(DateTime)
    
    def __repr__(self):
        return f"<UserSession(id={self.session_id}, jobs={self.total_jobs})>"


class CostTracking(Base):
    __tablename__ = 'cost_tracking'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time period
    date = Column(DateTime, index=True)
    hour = Column(Integer)  # 0-23
    
    # Resource usage
    gpu_type = Column(String(20))  # T4, V100, A100
    gpu_hours = Column(Float)
    cpu_hours = Column(Float)
    
    # Costs (Assignment: per-minute pricing)
    gpu_cost_per_hour = Column(Float)
    total_gpu_cost = Column(Float)
    storage_cost = Column(Float)
    network_cost = Column(Float)
    total_cost = Column(Float)
    
    # Volume
    jobs_processed = Column(Integer)
    minutes_generated = Column(Float)  # Total video minutes
    cost_per_minute = Column(Float)  # Key metric for assignment
    
    # Scaling (Assignment: 1-100 concurrent sessions)
    concurrent_sessions_avg = Column(Float)
    concurrent_sessions_peak = Column(Integer)
    instances_used = Column(Integer)
    
    # Timestamps
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    def __repr__(self):
        return f"<CostTracking(date={self.date}, cost=${self.total_cost:.4f})>"


# ============================================
# Database Helper Functions
# ============================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os


def get_database_url():
    return os.getenv('DATABASE_URL', 'sqlite:///./data/avatar_system.db')


def init_database():
    database_url = get_database_url()
    
    # Create engine
    engine = create_engine(
        database_url,
        echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    return engine


def get_session():
    engine = init_database()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# Usage example
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    engine = init_database()
    print(f" Database initialized: {get_database_url()}")
    
    # Create sample job
    session = get_session()
    
    job = GenerationJob(
        job_id="test-123",
        audio_path="test.wav",
        image_path="test.jpg",
        status="completed",
        fps_achieved=42.5,
        meets_realtime=True,
        quality_score=87.3,
        gpu_type="T4",
        compute_cost_usd=0.0006
    )
    
    session.add(job)
    session.commit()
    
    print(f" Sample job created: {job}")
    
    # Query
    jobs = session.query(GenerationJob).filter(GenerationJob.meets_realtime == True).all()
    print(f" Real-time jobs: {len(jobs)}")
    
    session.close()
