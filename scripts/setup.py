#!/usr/bin/env python3
"""
Setup script for Vietnamese Sign Language Detection System.
This script helps with project setup, dependency installation, and environment configuration.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def run_command(command, shell=True, check=True):
    """Run a command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def check_system_dependencies():
    """Check system-level dependencies."""
    print("Checking system dependencies...")
    
    # Check for camera
    if platform.system() == "Linux":
        if not os.path.exists("/dev/video0"):
            print("Warning: No camera detected at /dev/video0")
    
    # Check for audio
    try:
        if platform.system() == "Windows":
            run_command("where pulseaudio", check=False)
        else:
            run_command("which pulseaudio", check=False)
    except:
        print("Warning: PulseAudio not found (required for TTS)")
    
    print("System dependency check completed")


def setup_virtual_environment():
    """Setup Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    result = run_command(f"{sys.executable} -m venv venv")
    
    if result and result.returncode == 0:
        print("Virtual environment created successfully")
        return True
    else:
        print("Failed to create virtual environment")
        return False


def install_dependencies(dev=False):
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    # Determine the correct pip command
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip setuptools wheel")
    
    # Install requirements
    if dev:
        req_file = "requirements/dev.txt"
        print("Installing development dependencies...")
    else:
        req_file = "requirements/base.txt"
        print("Installing base dependencies...")
    
    result = run_command(f"{pip_cmd} install -r {req_file}")
    
    if result and result.returncode == 0:
        print("Dependencies installed successfully")
        return True
    else:
        print("Failed to install dependencies")
        return False


def create_directories():
    """Create required directories."""
    print("Creating required directories...")
    
    directories = [
        "Dataset/Video",
        "Dataset/Text", 
        "Data",
        "Models/checkpoints",
        "Logs",
        "Voice",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    print("Directory structure created")


def install_package():
    """Install the package in development mode."""
    print("Installing package in development mode...")
    
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    result = run_command(f"{pip_cmd} install -e .")
    
    if result and result.returncode == 0:
        print("Package installed successfully")
        return True
    else:
        print("Failed to install package")
        return False


def setup_pre_commit():
    """Setup pre-commit hooks."""
    print("Setting up pre-commit hooks...")
    
    if platform.system() == "Windows":
        precommit_cmd = "venv\\Scripts\\pre-commit"
    else:
        precommit_cmd = "venv/bin/pre-commit"
    
    result = run_command(f"{precommit_cmd} install")
    
    if result and result.returncode == 0:
        print("Pre-commit hooks installed successfully")
        return True
    else:
        print("Failed to install pre-commit hooks")
        return False


def create_example_config():
    """Create example configuration files."""
    print("Creating example configuration...")
    
    # Copy default config if it doesn't exist
    config_file = Path("configs/config.yaml")
    if not config_file.exists():
        print("Default config already exists")
        return
    
    # Create local config example
    local_config = Path("configs/config.local.yaml")
    if not local_config.exists():
        with open(local_config, 'w', encoding='utf-8') as f:
            f.write("""# Local configuration override
# Copy from config.yaml and modify as needed

# Example: Use different camera
# camera:
#   camera_index: 1

# Example: Enable debug logging
# logging:
#   level: "DEBUG"

# Example: Custom model path
# model:
#   model_path: "custom_models/my_model.keras"
""")
        print("Created example local config")


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    if platform.system() == "Windows":
        pytest_cmd = "venv\\Scripts\\pytest"
    else:
        pytest_cmd = "venv/bin/pytest"
    
    result = run_command(f"{pytest_cmd} tests/ -v")
    
    if result and result.returncode == 0:
        print("All tests passed ✓")
        return True
    else:
        print("Some tests failed")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup VSL Detection System")
    parser.add_argument("--dev", action="store_true", help="Setup for development")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment setup")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    
    args = parser.parse_args()
    
    print("=== Vietnamese Sign Language Detection System Setup ===")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system dependencies
    check_system_dependencies()
    
    # Create directories
    create_directories()
    
    # Setup virtual environment
    if not args.no_venv:
        if not setup_virtual_environment():
            sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(dev=args.dev):
        sys.exit(1)
    
    # Install package
    if not install_package():
        sys.exit(1)
    
    # Setup pre-commit for development
    if args.dev:
        setup_pre_commit()
    
    # Create example config
    create_example_config()
    
    # Run tests if requested
    if args.test:
        run_tests()
    
    print("\n=== Setup completed successfully! ===")
    
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    
    print("\nTo run the application:")
    print("  vsl-detect")
    
    print("\nTo run with custom config:")
    print("  vsl-detect --config configs/config.local.yaml")


if __name__ == "__main__":
    main()