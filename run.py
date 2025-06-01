#!/usr/bin/env python3
"""
Simple launcher script for Vietnamese Sign Language Detection System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Lỗi: Cần Python 3.8 trở lên")
        print(f"   Phiên bản hiện tại: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Kiểm tra dependencies...")
    
    required_packages = [
        'PyQt5', 'opencv-python', 'tensorflow', 
        'mediapipe', 'numpy', 'gtts', 'pygame'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').lower())
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    return missing

def install_dependencies():
    """Install missing dependencies."""
    print("📦 Cài đặt dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies đã được cài đặt thành công!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt dependencies: {e}")
        return False

def run_application():
    """Run the main application."""
    print("🚀 Khởi chạy ứng dụng...")
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Run the main application
        subprocess.run([sys.executable, 'src/vsl_detect/main.py'])
    except Exception as e:
        print(f"❌ Lỗi khởi chạy ứng dụng: {e}")

def main():
    """Main launcher function."""
    print("🇻🇳 Vietnamese Sign Language Detection System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Thiếu {len(missing)} packages cần thiết")
        response = input("Có muốn cài đặt tự động không? (y/n): ").lower().strip()
        
        if response in ['y', 'yes', 'có']:
            if not install_dependencies():
                return
        else:
            print("💡 Hãy chạy: pip install -r requirements.txt")
            return
    
    print("\n🎯 Tất cả dependencies đã sẵn sàng!")
    print("📋 Các tính năng có sẵn:")
    print("   🖥️  GUI Application - Giao diện chính")
    print("   🧠  AI Detection - Nhận dạng ngôn ngữ ký hiệu")
    print("   🔊  Text-to-Speech - Chuyển đổi văn bản thành giọng nói")
    print("   📹  Real-time Processing - Xử lý video thời gian thực")
    
    input("\n👆 Nhấn Enter để khởi chạy ứng dụng...")
    run_application()

if __name__ == "__main__":
    main()