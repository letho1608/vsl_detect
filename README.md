# Vietnamese Sign Language Detection

> Advanced Vietnamese Sign Language Detection using Computer Vision and Deep Learning

## 🚀 Quick Start

### Cách đơn giản nhất (Khuyến nghị)
```bash
# Chạy script launcher tự động
python run.py
```

### Hoặc sử dụng menu tương tác
```bash
# Menu với nhiều tùy chọn
python quick_start.py
```

### Cách thủ công
```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy ứng dụng GUI
python src/vsl_detect/main.py
```

## 🎯 Features

- 🧠 **AI-Powered Recognition**: Deep learning model for accurate sign detection
- ⚡ **Real-time Processing**: Optimized for live video detection
- 🖥️ **User-friendly Interface**: Easy-to-use GUI application
- 📈 **Data Optimization**: Advanced data augmentation tools
- 🚀 **Easy Launcher**: Simple scripts to run project with one command
- 🎛️ **Interactive Menu**: User-friendly menu for all project functions

## 📁 Project Structure

```
vsl_detect/
├── run.py               # 🚀 Simple launcher script (Khuyến nghị)
├── quick_start.py       # 🎛️ Interactive menu launcher
├── requirements.txt     # 📋 Core dependencies
├── README.md            # 📖 This file
├── LICENSE              # 📄 MIT License
├── src/vsl_detect/      # 📦 Source code modules
│   ├── main.py          # 🖥️ Main GUI application
│   ├── core/            # 🧠 Core detection modules
│   ├── data/            # 📊 Data processing
│   ├── ui/              # 🖥️ UI components
│   └── utils/           # 🔧 Utilities
├── apps/                # 📊 Training & data apps
│   ├── Training.py      # 🧠 Model training script
│   └── CreateData.py    # 📊 Data preparation script
├── tools/               # ⚡ Optimization tools
├── configs/             # ⚙️ Configuration files
├── requirements/        # 📋 Detailed requirements
├── dev/                 # 🔧 Development files
├── tests/               # 🧪 Test files
├── scripts/             # 🔧 Utility scripts
├── Logs/                # 📊 Training logs
└── Models/              # 🤖 Trained models
```

## 🔧 Usage

### Các cách chạy ứng dụng

#### 🎯 Sử dụng Script Launcher (Khuyến nghị)
```bash
python run.py
```

#### 🎛️ Sử dụng Menu Tương Tác
```bash
python quick_start.py
```

#### ⚙️ Chạy trực tiếp từng thành phần
```bash
# Ứng dụng GUI chính
python src/vsl_detect/main.py

# Huấn luyện mô hình
python apps/Training.py

# Tạo dữ liệu
python apps/CreateData.py
```

### Advanced Usage
```bash
# Development setup
pip install -r requirements/dev.txt

# Run tests
python -m pytest tests/

# Production setup
pip install -r requirements/prod.txt

# Run with arguments
python src/vsl_detect/main.py --config configs/config.yaml --debug
```

## 🚀 Launcher Scripts

### `run.py` - Simple Launcher
- Tự động kiểm tra Python version (cần 3.8+)
- Kiểm tra và cài đặt dependencies nếu thiếu
- Khởi chạy ứng dụng GUI chính
- **Khuyến nghị sử dụng cho người dùng mới**

### `quick_start.py` - Interactive Menu
- Menu tương tác với 5 tùy chọn:
  1. 🖥️ Chạy ứng dụng GUI chính
  2. 🧠 Huấn luyện mô hình AI
  3. 📊 Tạo và chuẩn bị dữ liệu
  4. 📦 Cài đặt dependencies
  5. 🧪 Chạy tests
- **Khuyến nghị cho developers và power users**

## 📊 Performance

- **Accuracy**: 85-95% (with data optimization)
- **Speed**: Real-time processing (30+ FPS)
- **Requirements**: Python 3.8+, 8GB RAM recommended

## 🛠️ Development

### Docker Deployment
```bash
# Build and run with Docker
docker build -f dev/Dockerfile -t vsl-detect .
docker run -p 8080:8080 vsl-detect

# Or use docker-compose
cd dev/
docker-compose up
```

### Build Commands
```bash
# Using Makefile
make install      # Install dependencies
make test         # Run tests
make build        # Build package
make docker       # Build Docker image

# Manual commands
cd dev/
python setup.py build
python setup.py install
```

## 📖 Documentation

- **[Tools Documentation](tools/)** - Optimization and utility tools
- **[Configuration](configs/)** - Configuration files
- **[Development Guide](dev/)** - Development setup and build instructions
- **[Requirements](requirements/)** - Detailed dependency specifications

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

**Made with ❤️ for Vietnamese Sign Language community** 🇻🇳
