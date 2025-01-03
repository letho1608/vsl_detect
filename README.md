# Sign Language Recognition System App
A real-time sign language recognition system using deep learning and computer vision techniques.

## Project Structure

```
├── Data/                  # Processed data directory
├── Dataset/              # Raw video data and labels
├── Logs/                # Logs and mapping directory
├── Models/              # Trained models directory
│   └── checkpoints/    # Training checkpoints
└── Scripts/
    ├── CreateData.py   # Data collection and processing
    ├── CheckData.py    # Data integrity verification
    ├── Training.py     # Model training
    └── Main.py         # Main recognition program
```

## System Requirements

### Hardware
- CPU or CUDA-capable GPU (recommended)
- Webcam for real-time recognition
- Minimum 8GB RAM

### Software
- Python 3.6 or higher
- PyTorch
- MediaPipe
- OpenCV
- CUDA Toolkit (optional)

## Installation

1. Clone repository:
```bash
git clone <repository-url>
cd sign-language-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```bash
python CreateData.py
```
- Collects keypoints from videos
- Processes and saves data automatically
- Creates mapping between original and processed names

### 2. Data Verification
```bash
python CheckData.py
```
- Checks data integrity
- Auto-fixes common issues
- Generates detailed statistics

### 3. Model Training
```bash
python Training.py
```
- Uses LSTM model with attention mechanism
- Supports checkpointing and training resumption
- Visual progress tracking with colors

### 4. Recognition
```bash
python Main.py
```
- Real-time recognition through webcam
- Displays results and confidence scores
- Supports multiple gestures

## Features

- [x] Automatic video processing
- [x] MediaPipe keypoint extraction
- [x] Smart action name mapping
- [x] Training checkpoints
- [x] Visual progress tracking
- [x] Real-time recognition
- [x] Confidence evaluation

## Notes

- Ensure sufficient disk space for training data
- Check CUDA compatibility for GPU usage
- Adjust batch size according to RAM/VRAM

## Contributing

Contributions are welcome. Please create an issue or pull request.

## License

© 2024 Sign Language Recognition System
