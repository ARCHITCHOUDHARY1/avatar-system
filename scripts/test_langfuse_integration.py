"""
Test Langfuse Integration with Avatar Pipeline
Runs a simple pipeline test with Langfuse tracing
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.observability import (
    is_langfuse_enabled,
    LangfuseMonitor,
    trace_function,
)

logger = logging.getLogger(__name__)


@trace_function(name="test_preprocessing", metadata={"version": "1.0"})
def test_preprocessing():
    """Test function with tracing"""
    import time
    logger.info("Running preprocessing...")
    time.sleep(0.5)
    return {"status": "success", "data": "preprocessed"}


@trace_function(name="test_processing", metadata={"version": "1.0"})
def test_processing():
    """Test function with tracing"""
    import time
    logger.info("Running processing...")
    time.sleep(1.0)
    return {"status": "success", "data": "processed"}


def test_langfuse_integration():
    """Test Langfuse integration with mock pipeline"""
    print("=" * 60)
    print("LANGFUSE INTEGRATION TEST")
    print("=" * 60)
    
    if not is_langfuse_enabled():
        print("\n??  Langfuse is disabled")
        print("   Set ENABLE_LANGFUSE=true in .env to enable")
        return False
    
    try:
        # Create monitor
        monitor = LangfuseMonitor(session_id="test_session_123")
        
        print("\n? Starting test session...")
        monitor.start_session(
            metadata={
                "test": True,
                "pipeline": "mock_pipeline",
                "version": "1.0"
            }
        )
        
        # Test node tracking
        print("? Testing node tracking...")
        monitor.log_node_start("preprocessing", metadata={"step": 1})
        result1 = test_preprocessing()
        monitor.log_node_end("preprocessing", output=result1)
        
        monitor.log_node_start("processing", metadata={"step": 2})
        result2 = test_processing()
        monitor.log_node_end("processing", output=result2)
        
        # Test metrics logging
        print("? Testing metrics logging...")
        monitor.log_metrics({
            "total_duration": 1.5,
            "nodes_executed": 2,
            "success_rate": 1.0,
        })
        
        # Test quality scores
        print("? Testing quality scores...")
        monitor.create_score(
            name="test_quality",
            value=0.95,
            comment="Test quality score"
        )
        
        # End session
        print("? Ending session...")
        monitor.end_session(
            final_metrics={
                "total_nodes": 2,
                "total_duration": 1.5,
                "status": "success"
            }
        )
        
        print("\n" + "=" * 60)
        print("? LANGFUSE INTEGRATION TEST PASSED")
        print("=" * 60)
        print("\nCheck your Langfuse dashboard to see:")
        print("  - Session trace: test_session_123")
        print("  - 2 node executions (preprocessing, processing)")
        print("  - Performance metrics")
        print("  - Quality score")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n? Integration test failed: {e}")
        logger.exception("Test failed")
        return False


if __name__ == "__main__":
    success = test_langfuse_integration()
    sys.exit(0 if success else 1)
