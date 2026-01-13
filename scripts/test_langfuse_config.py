"""
Test Langfuse Configuration
Verifies that Langfuse is properly configured and can connect
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_config,
)

def test_langfuse_config():
    """Test Langfuse configuration"""
    print("=" * 60)
    print("LANGFUSE CONFIGURATION TEST")
    print("=" * 60)
    
    # Check if enabled
    enabled = is_langfuse_enabled()
    print(f"\n? Langfuse Enabled: {enabled}")
    
    if not enabled:
        print("\n??  Langfuse is disabled")
        print("   Set ENABLE_LANGFUSE=true in .env to enable")
        return False
    
    # Check configuration
    config = get_config()
    print(f"\n? Configuration: {config}")
    
    # Check client
    client = get_langfuse_client()
    if client is None:
        print("\n? Failed to create Langfuse client")
        print("   Check your API keys in .env:")
        print("   - LANGFUSE_PUBLIC_KEY")
        print("   - LANGFUSE_SECRET_KEY")
        print("   - LANGFUSE_HOST")
        return False
    
    print(f"\n? Client created successfully")
    print(f"   Host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    
    # Test connection
    try:
        print("\n? Testing connection...")
        # Create a test trace
        trace = client.trace(
            name="test_trace",
            metadata={"test": True},
        )
        print("? Test trace created successfully")
        
        # Flush
        client.flush()
        print("? Traces flushed successfully")
        
        print("\n" + "=" * 60)
        print("? LANGFUSE CONFIGURATION TEST PASSED")
        print("=" * 60)
        print("\nYou can now view traces at:")
        print(f"   {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n? Connection test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your API keys are correct")
        print("2. Check your internet connection")
        print("3. Ensure LANGFUSE_HOST is accessible")
        print("4. Try setting LANGFUSE_DEBUG=true for more details")
        return False


if __name__ == "__main__":
    success = test_langfuse_config()
    sys.exit(0 if success else 1)
