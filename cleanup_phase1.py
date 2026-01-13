#!/usr/bin/env python3
"""
Automated cleanup script for Avatar System Orchestrator
Removes unnecessary files identified in cleanup plan
"""

import os
from pathlib import Path

# Root directory
ROOT = Path(__file__).parent

# Files to remove - Phase 1
FILES_TO_REMOVE = [
    # Test files (keep only test_generation_api.py)
    "test_system.py",
    "test_sadtalker.py",
    "test_import.py",
    "test_deps.py",
    "debug_server.py",
    
    # Temporary output files
    "debug_output.txt",
    "structure_output.txt",
    "voice_generation_output.txt",
    "voice_input.wav",
    
    # Colab-specific files
    "colab_notebook.ipynb",
    "colab_runner.py",
    "example_with_langfuse.py",
    "check_langsmith.py",
    
    # Duplicate download scripts (will merge later)
    "download_sadtalker_checkpoints.py",
    "setup_sadtalker.py",
    
    # Duplicate documentation (will merge into OBSERVABILITY.md)
    "LANGFUSE_DATASETS.md",
    "LANGFUSE_INTEGRATION.md",
    "LANGFUSE_QUICKSTART.md",
    "LANGFUSE_SETUP.md",
    "LANGFUSE_SUMMARY.md",
    "LANGSMITH_INTEGRATION.md",
    "GIT_PUSH_INSTRUCTIONS.md",
]

# Directories to remove completely
DIRS_TO_REMOVE = [
    "notebooks",  # Empty or unused
]

def remove_files():
    """Remove unnecessary files"""
    print("=" * 60)
    print("PHASE 1: File Cleanup")
    print("=" * 60)
    
    removed = []
    not_found = []
    
    for file in FILES_TO_REMOVE:
        file_path = ROOT / file
        if file_path.exists():
            try:
                file_path.unlink()
                removed.append(file)
                print(f"✓ Removed: {file}")
            except Exception as e:
                print(f"✗ Failed to remove {file}: {e}")
        else:
            not_found.append(file)
            print(f"⊘ Not found: {file}")
    
    print(f"\nSummary:")
    print(f"  Removed: {len(removed)} files")
    print(f"  Not found: {len(not_found)} files")
    
    return removed

def remove_directories():
    """Remove unnecessary directories"""
    import shutil
    
    print("\n" + "=" * 60)
    print("Removing unused directories")
    print("=" * 60)
    
    for dir_name in DIRS_TO_REMOVE:
        dir_path = ROOT / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Removed directory: {dir_name}")
            except Exception as e:
                print(f"✗ Failed to remove {dir_name}: {e}")
        else:
            print(f"⊘ Not found: {dir_name}")

if __name__ == "__main__":
    print("\n🧹 Avatar System Cleanup Script")
    print("Removing unnecessary files...\n")
    
    removed_files = remove_files()
    remove_directories()
    
    print("\n" + "=" * 60)
    print(f"✅ Cleanup complete! Removed {len(removed_files)} files")
    print("=" * 60)
