#!/usr/bin/env python3
"""
Quick start menu for Vietnamese Sign Language Detection System
"""

import os
import sys
import subprocess
from pathlib import Path

def show_menu():
    """Display main menu options."""
    print("\n🇻🇳 Vietnamese Sign Language Detection System")
    print("=" * 55)
    print("📋 Chọn chức năng:")
    print()
    print("1️⃣  Chạy ứng dụng GUI chính")
    print("2️⃣  Huấn luyện mô hình AI")
    print("3️⃣  Tạo và chuẩn bị dữ liệu")
    print("4️⃣  Cài đặt dependencies")
    print("5️⃣  Chạy tests")
    print("0️⃣  Thoát")
    print()

def run_gui():
    """Run the main GUI application."""
    print("🖥️  Đang khởi chạy ứng dụng GUI...")
    try:
        subprocess.run([sys.executable, 'src/vsl_detect/main.py'])
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def run_training():
    """Run model training."""
    print("🧠 Đang khởi chạy quá trình huấn luyện...")
    try:
        subprocess.run([sys.executable, 'apps/Training.py'])
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def run_data_creation():
    """Run data creation process."""
    print("📊 Đang khởi chạy quá trình tạo dữ liệu...")
    try:
        subprocess.run([sys.executable, 'apps/CreateData.py'])
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def install_dependencies():
    """Install project dependencies."""
    print("📦 Đang cài đặt dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies đã được cài đặt thành công!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt: {e}")

def run_tests():
    """Run project tests."""
    print("🧪 Đang chạy tests...")
    try:
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def main():
    """Main menu loop."""
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    while True:
        show_menu()
        
        try:
            choice = input("👉 Nhập lựa chọn (0-5): ").strip()
            
            if choice == '1':
                run_gui()
            elif choice == '2':
                run_training()
            elif choice == '3':
                run_data_creation()
            elif choice == '4':
                install_dependencies()
            elif choice == '5':
                run_tests()
            elif choice == '0':
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ. Vui lòng chọn từ 0-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        input("\n👆 Nhấn Enter để tiếp tục...")

if __name__ == "__main__":
    main()