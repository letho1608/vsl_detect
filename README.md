# Vietnamese Sign Language Recognition System

A real-time sign language recognition system using deep learning and computer vision techniques, specialized for Vietnamese sign language.

## Directory Structure

```
├── Dataset/                # Raw data directory
│   ├── Video/             # Original sign language videos
│   └── Text/              # Text labels (Label.csv)
├── Data/                  # Processed training data
├── Models/                # Trained model files
│   ├── checkpoints/      # Training checkpoints
│   └── final_model.keras # Final trained model
├── Logs/                 # Logs and mappings
└── Scripts/
    ├── Main.py          # Main application
    ├── Combine.py      # Video downloader and create label
    ├── CreateData.py    # Data processor
    ├── Training.py      # Model trainer
    └── CheckData.py     # Data integrity checker
```

## Requirements

### Hardware
- Webcam for real-time recognition
- 8GB RAM minimum
- CUDA-capable GPU (recommended)

### Software
- Python 3.8+
- TensorFlow 2.x
- MediaPipe
- OpenCV
- PyQt5
- CUDA Toolkit (optional)

## Installation

1. Install required software:
```bash
# Install Python dependencies
pip install tensorflow opencv-python mediapipe PyQt5 
pip install gtts playsound pandas tqdm pygame pywin32
```

2. Clone and setup:
```bash
git clone [<repository-url>](https://github.com/letho1608/vsl_detect)
cd vsl-detect
```

## Usage

### 1. Data Collection
```bash
# Download videos
python Combine.py
```

### 2. Data Processing
```bash
# Process videos into training data
python CreateData.py

# Verify data integrity
python CheckData.py
```

### 3. Model Training
```bash
python Training.py
```

### 4. Recognition
```bash
python Main.py
```

## Features

- [x] Automated video download
- [x] Vietnamese text extraction
- [x] MediaPipe landmark extraction
- [x] Real-time recognition
- [x] Text-to-speech output
- [x] Modern PyQt5 GUI
- [x] Training progress visualization
- [x] Checkpoint system
- [x] Data integrity checking

## System Components

1. **Video Downloader**
   - Downloads sign language videos
   - Maintains video quality
   - Handles connection errors

2. **Text Extractor**
   - Extracts Vietnamese text from videos
   - Creates Label.csv

3. **Data Processor**
   - Extracts MediaPipe landmarks
   - Processes video sequences
   - Generates training data

4. **Model Trainer**
   - LSTM-based architecture
   - Checkpoint system
   - Progress visualization

5. **Main Application**
   - Real-time recognition
   - Vietnamese text-to-speech
   - User-friendly interface

## Notes

- GPU acceleration requires compatible NVIDIA drivers
- Maintain stable internet connection for video downloads
- Check storage space for training data

## Troubleshooting

- Check Tesseract installation if text extraction fails
- Verify CUDA setup for GPU acceleration
- Monitor logs in Logs directory
- Ensure proper video format (MP4)

## License

© 2024 Vietnamese Sign Language Recognition System. All rights reserved.
