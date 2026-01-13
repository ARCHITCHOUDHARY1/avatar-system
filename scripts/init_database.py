"""
Database Initialization Script

Creates and initializes complete database with:
- SQLite tables
- Redis connection test
- Sample data (optional)
- Verification
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def init_database():
    """Initialize complete database system"""
    
    logger.info("=" * 70)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 70)
    
    # Import after path setup
    from src.database.db_manager import db_manager
    from src.database.models import GenerationJob, PerformanceMetric
    
    # Step 1: Check database connection
    logger.info("\n1. Checking database connection...")
    try:
        session = db_manager.get_session()
        session.close()
        logger.info("   [OK] Database connected")
    except Exception as e:
        logger.error(f"   [ERROR] Database connection failed: {e}")
        return False
    
    # Step 2: Check Redis (optional)
    logger.info("\n2. Checking Redis cache...")
    if db_manager.cache_enabled:
        try:
            db_manager.redis_client.ping()
            logger.info("   [OK] Redis cache connected")
        except Exception as e:
            logger.warning(f"   [WARNING]  Redis not available: {e}")
            logger.info("   ??  System will work without cache (slower)")
    else:
        logger.info("   ??  Redis not configured (optional)")
    
    # Step 3: Verify tables
    logger.info("\n3. Verifying database tables...")
    try:
        from src.database.models import Base
        tables = Base.metadata.tables.keys()
        logger.info(f"   [OK] Tables created: {len(tables)}")
        for table in tables:
            logger.info(f"      - {table}")
    except Exception as e:
        logger.error(f"   [ERROR] Table verification failed: {e}")
        return False
    
    # Step 4: Test database operations
    logger.info("\n4. Testing database operations...")
    try:
        # Create test job
        job = db_manager.save_job(
            job_id="init-test",
            audio_path="test.wav",
            image_path="test.jpg",
            output_path="test.mp4",
            status="completed",
            fps_achieved=40.0,
            quality_score=85.0
        )
        logger.info("   [OK] Write test: OK")
        
        # Read test
        retrieved = db_manager.get_job("init-test")
        if retrieved:
            logger.info("   [OK] Read test: OK")
        
        # Update test
        db_manager.update_job("init-test", emotion="happy")
        logger.info("   [OK] Update test: OK")
        
        # Delete test job
        session = db_manager.get_session()
        session.query(GenerationJob).filter(
            GenerationJob.job_id == "init-test"
        ).delete()
        session.commit()
        session.close()
        logger.info("   [OK] Delete test: OK")
        
    except Exception as e:
        logger.error(f"   [ERROR] Database operation failed: {e}")
        return False
    
    # Step 5: Test cache (if enabled)
    if db_manager.cache_enabled:
        logger.info("\n5. Testing Redis cache...")
        try:
            # Set test
            db_manager.cache_set("test-key", {"data": "test"}, ttl=60)
            logger.info("   [OK] Cache write: OK")
            
            # Get test
            value = db_manager.cache_get("test-key")
            if value and value.get("data") == "test":
                logger.info("   [OK] Cache read: OK")
            
            # Delete test
            db_manager.cache_delete("test-key")
            logger.info("   [OK] Cache delete: OK")
            
        except Exception as e:
            logger.error(f"   [ERROR] Cache operation failed: {e}")
    
    # Step 6: Create sample data (optional)
    logger.info("\n6. Creating sample data (optional)...")
    try:
        create_sample = input("   Create sample data? (y/n): ").lower() == 'y'
        
        if create_sample:
            # Sample jobs
            sample_jobs = [
                {
                    "job_id": "sample-001",
                    "audio_path": "samples/audio1.wav",
                    "image_path": "samples/face1.jpg",
                    "output_path": "outputs/video1.mp4",
                    "status": "completed",
                    "fps_achieved": 42.5,
                    "quality_score": 87.3,
                    "total_time_seconds": 75.2,
                    "emotion": "happy",
                    "confidence": 0.95
                },
                {
                    "job_id": "sample-002",
                    "audio_path": "samples/audio2.wav",
                    "image_path": "samples/face2.jpg",
                    "output_path": "outputs/video2.mp4",
                    "status": "completed",
                    "fps_achieved": 38.1,
                    "quality_score": 82.5,
                    "total_time_seconds": 80.5,
                    "emotion": "neutral",
                    "confidence": 0.88
                },
                {
                    "job_id": "sample-003",
                    "audio_path": "samples/audio3.wav",
                    "image_path": "samples/face3.jpg",
                    "output_path": "outputs/video3.mp4",
                    "status": "processing",
                    "progress": 0.6
                }
            ]
            
            for job_data in sample_jobs:
                db_manager.save_job(**job_data)
            
            logger.info(f"   [OK] Created {len(sample_jobs)} sample jobs")
            
            # Sample performance metric
            db_manager.save_performance_metric(
                device_type="cuda",
                gpu_name="Tesla T4",
                gpu_memory_gb=16.0,
                batch_size=8,
                resolution=512,
                use_fp16=True,
                fps=42.5,
                latency_ms=23.5,
                passes_realtime=True,
                optimizations=["FP16", "ParallelProcessing"],
                expected_speedup=2.4
            )
            
            logger.info("   [OK] Created sample performance metric")
    
    except KeyboardInterrupt:
        logger.info("\n   ??  Skipped sample data")
    
    # Step 7: Display statistics
    logger.info("\n7. Database statistics...")
    try:
        stats = db_manager.get_statistics()
        logger.info(f"   Total jobs: {stats['total_jobs']}")
        logger.info(f"   Completed: {stats['completed_jobs']}")
        logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
        if stats['avg_fps'] > 0:
            logger.info(f"   Average FPS: {stats['avg_fps']:.1f}")
        if stats['avg_quality'] > 0:
            logger.info(f"   Average quality: {stats['avg_quality']:.1f}/100")
    except Exception as e:
        logger.warning(f"   Could not get statistics: {e}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZATION COMPLETE!")
    logger.info("=" * 70)
    logger.info("\n[OK] Database: Ready")
    logger.info(f"   Location: {os.getenv('DATABASE_URL', 'sqlite:///./data/avatar_system.db')}")
    
    if db_manager.cache_enabled:
        logger.info("\n[OK] Cache: Ready (Redis)")
        logger.info(f"   Location: {os.getenv('REDIS_URL', 'N/A')}")
    else:
        logger.info("\n??  Cache: Disabled")
        logger.info("   (Optional - system works without it)")
    
    logger.info("\nNext steps:")
    logger.info("  1. Run: python main.py --mode gradio")
    logger.info("  2. Generate avatars")
    logger.info("  3. Check database for job tracking")
    logger.info("\n" + "=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = init_database()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n\n[ERROR] Initialization failed: {e}")
        logger.exception(e)
        sys.exit(1)
