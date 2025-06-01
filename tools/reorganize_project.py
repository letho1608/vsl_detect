#!/usr/bin/env python3
"""
Reorganize Vietnamese Sign Language Detection project structure.
Đổi tên files/folders và đặt vào vị trí phù hợp theo chuẩn professional.
"""

import os
import shutil
from pathlib import Path
import json

def create_directory_structure():
    """Create proper directory structure."""
    print("📁 CREATING PROPER DIRECTORY STRUCTURE")
    print("-" * 50)
    
    directories = [
        "vsl_detection",           # Main source package
        "vsl_detection/core",      # Core detection modules
        "vsl_detection/models",    # Model definitions
        "vsl_detection/data",      # Data processing
        "vsl_detection/utils",     # Utilities
        "vsl_detection/gui",       # GUI components
        "scripts",                 # Scripts and tools
        "docs",                   # Documentation
        "tests",                  # Tests
        "configs",                # Configuration files
        "requirements",           # Dependencies
        "models",                 # Trained models storage
        "examples",               # Usage examples
        ".github",                # GitHub specific files
        ".github/workflows"       # GitHub Actions
    ]
    
    created_count = 0
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
        created_count += 1
    
    print(f"📊 Created {created_count} directories")

def main():
    """Main reorganization function."""
    print("🔄 VIETNAMESE SIGN LANGUAGE DETECTION - PROJECT REORGANIZATION")
    print("🎯 This tool has been moved to tools/ directory")
    print("=" * 80)
    
    print("✅ Tool successfully relocated to maintain clean root directory")

if __name__ == "__main__":
    main()