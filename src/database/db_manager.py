
import os
import logging
from typing import Optional, Any, Callable, Dict, List
from datetime import datetime
import json
import hashlib
from pathlib import Path

# SQLAlchemy
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Models
from src.database.models import (
    Base,
    GenerationJob,
    PerformanceMetric,
    QualityReport,
    UserSession,
    CostTracking
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    
    def __init__(self, database_url: Optional[str] = None, redis_url: Optional[str] = None):
        # Database setup
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'sqlite:///./data/avatar_system.db'
        )
        
        # Create engine
        if self.database_url.startswith('sqlite'):
            # SQLite specific settings
            self.engine = create_engine(
                self.database_url,
                connect_args={'check_same_thread': False},
                poolclass=StaticPool,
                echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
            )
        else:
            # PostgreSQL/MySQL settings
            self.engine = create_engine(
                self.database_url,
                pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '10')),
                echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables
        self._create_tables()
        
        # Redis setup (optional)
        self.redis_url = redis_url or os.getenv('REDIS_URL')
        self.redis_client = None
        self.cache_enabled = False
        
        if self.redis_url and REDIS_AVAILABLE:
            self._setup_redis()
        else:
            logger.info("Redis not configured - caching disabled")
        
        logger.info(f"[OK] Database manager initialized")
        logger.info(f"   Database: {self._mask_password(self.database_url)}")
        logger.info(f"   Cache: {'Enabled (Redis)' if self.cache_enabled else 'Disabled'}")
    
    def _create_tables(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("[OK] Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def _setup_redis(self):
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("[OK] Redis cache connected")
            
        except Exception as e:
            logger.warning(f"[WARNING] Redis connection failed: {e}")
            logger.warning("   Continuing without cache")
            self.redis_client = None
            self.cache_enabled = False
    
    def _mask_password(self, url: str) -> str:
        if '://' in url and '@' in url:
            protocol, rest = url.split('://', 1)
            if '@' in rest:
                creds, host = rest.split('@', 1)
                if ':' in creds:
                    user, _ = creds.split(':', 1)
                    return f"{protocol}://{user}:****@{host}"
        return url
    
    def get_session(self) -> Session:
        return self.SessionLocal()
    
    # ============================================
    # Cache Operations (Redis)
    # ============================================
    
    def cache_get(self, key: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def cache_set(self, key: str, value: Any, ttl: int = 3600):
        if not self.cache_enabled:
            return
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def cache_delete(self, key: str):
        if not self.cache_enabled:
            return
        
        try:
            self.redis_client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def make_cache_key(self, *args) -> str:
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached_or_generate(
        self, 
        key: str, 
        generator: Callable, 
        ttl: int = 3600
    ) -> Any:

        # Try cache first
        cached = self.cache_get(key)
        if cached is not None:
            return cached
        
        # Generate
        value = generator()
        
        # Cache for next time
        self.cache_set(key, value, ttl)
        
        return value
    
    # ============================================
    # Job Operations (SQLite/PostgreSQL)
    # ============================================
    
    def save_job(
        self,
        job_id: str,
        audio_path: str,
        image_path: str,
        output_path: str,
        **kwargs
    ) -> GenerationJob:
 
        session = self.get_session()
        
        try:
            job = GenerationJob(
                job_id=job_id,
                audio_path=audio_path,
                image_path=image_path,
                output_path=output_path,
                **kwargs
            )
            
            session.add(job)
            session.commit()
            session.refresh(job)
            
            logger.info(f"[OK] Job saved: {job_id}")
            
            return job
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save job: {e}")
            raise
        finally:
            session.close()
    
    def update_job(self, job_id: str, **updates) -> Optional[GenerationJob]:
        session = self.get_session()
        
        try:
            job = session.query(GenerationJob).filter(
                GenerationJob.job_id == job_id
            ).first()
            
            if not job:
                logger.warning(f"Job not found: {job_id}")
                return None
            
            for key, value in updates.items():
                setattr(job, key, value)
            
            job.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(job)
            
            logger.debug(f"Job updated: {job_id}")
            
            return job
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update job: {e}")
            raise
        finally:
            session.close()
    
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        session = self.get_session()
        
        try:
            job = session.query(GenerationJob).filter(
                GenerationJob.job_id == job_id
            ).first()
            return job
        finally:
            session.close()
    
    def get_recent_jobs(self, limit: int = 10) -> List[GenerationJob]:
        session = self.get_session()
        
        try:
            jobs = session.query(GenerationJob).order_by(
                GenerationJob.created_at.desc()
            ).limit(limit).all()
            return jobs
        finally:
            session.close()
    
    # ============================================
    # Performance Metrics
    # ============================================
    
    def save_performance_metric(self, **kwargs) -> PerformanceMetric:
        session = self.get_session()
        
        try:
            metric = PerformanceMetric(**kwargs)
            session.add(metric)
            session.commit()
            session.refresh(metric)
            
            logger.debug("Performance metric saved")
            
            return metric
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save metric: {e}")
            raise
        finally:
            session.close()
    
    # ============================================
    # Quality Reports
    # ============================================
    
    def save_quality_report(self, job_id: str, **kwargs) -> QualityReport:
        session = self.get_session()
        
        try:
            report = QualityReport(job_id=job_id, **kwargs)
            session.add(report)
            session.commit()
            session.refresh(report)
            
            logger.debug(f"Quality report saved for job: {job_id}")
            
            return report
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save quality report: {e}")
            raise
        finally:
            session.close()
    
    # ============================================
    # Statistics & Analytics
    # ============================================
    
    def get_statistics(self) -> Dict[str, Any]:
        session = self.get_session()
        
        try:
            total_jobs = session.query(GenerationJob).count()
            completed_jobs = session.query(GenerationJob).filter(
                GenerationJob.status == 'completed'
            ).count()
            
            avg_fps = session.query(GenerationJob.fps_achieved).filter(
                GenerationJob.fps_achieved.isnot(None)
            ).all()
            
            avg_quality = session.query(GenerationJob.quality_score).filter(
                GenerationJob.quality_score.isnot(None)
            ).all()
            
            stats = {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
                'avg_fps': sum(r[0] for r in avg_fps) / len(avg_fps) if avg_fps else 0,
                'avg_quality': sum(r[0] for r in avg_quality) / len(avg_quality) if avg_quality else 0
            }
            
            return stats
            
        finally:
            session.close()
    
    def cleanup_old_data(self, days: int = 30):
        session = self.get_session()
        
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            deleted = session.query(GenerationJob).filter(
                GenerationJob.created_at < cutoff,
                GenerationJob.status == 'completed'
            ).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {deleted} old jobs")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Cleanup failed: {e}")
        finally:
            session.close()


# ============================================
# Global Instance
# ============================================

db_manager = DatabaseManager()


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Save a job
    job = db_manager.save_job(
        job_id="test-123",
        audio_path="test.wav",
        image_path="test.jpg",
        output_path="output.mp4",
        status="completed",
        fps_achieved=42.5,
        quality_score=87.3,
        total_time_seconds=75.2
    )
    
    print(f"[OK] Job saved: {job.job_id}")
    
    # Update job
    db_manager.update_job("test-123", emotion="happy", confidence=0.95)
    print(f"[OK] Job updated")
    
    # Cache example
    def expensive_operation():
        print("Generating (expensive)...")
        return {"result": "avatar_data"}
    
    # First call: generates and caches
    cache_key = db_manager.make_cache_key("avatar", "test.wav", "test.jpg")
    result1 = db_manager.get_cached_or_generate(cache_key, expensive_operation)
    
    # Second call: instant from cache
    result2 = db_manager.get_cached_or_generate(cache_key, expensive_operation)
    
    # Get statistics
    stats = db_manager.get_statistics()
    print(f"\n[STATS] Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
