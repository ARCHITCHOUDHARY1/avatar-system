#!/bin/bash
# Git Push Guide for Avatar System Orchestrator
# Repository: https://github.com/ARCHITCHOUDHARY1/avatar-system.git

echo "=========================================="
echo "Git Push Guide - Avatar System"
echo "=========================================="

# Step 1: Check current git status
echo -e "\n[Step 1] Checking git status..."
git status

# Step 2: Stage all changes
echo -e "\n[Step 2] Staging all changes..."
git add .

# Step 3: Commit changes
echo -e "\n[Step 3] Committing changes..."
git commit -m "Cleanup Phase 1: Removed 22 files, fixed duplicates, updated docs"

# Step 4: Set main branch
echo -e "\n[Step 4] Setting main branch..."
git branch -M main

# Step 5: Add remote origin
echo -e "\n[Step 5] Adding remote origin..."
git remote add origin https://github.com/ARCHITCHOUDHARY1/avatar-system.git 2>/dev/null || 
git remote set-url origin https://github.com/ARCHITCHOUDHARY1/avatar-system.git

# Step 6: Push to GitHub
echo -e "\n[Step 6] Pushing to GitHub..."
git push -u origin main

echo -e "\n=========================================="
echo "✅ Push complete!"
echo "View at: https://github.com/ARCHITCHOUDHARY1/avatar-system"
echo "=========================================="
