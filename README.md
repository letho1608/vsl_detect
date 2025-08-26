# Vietnamese Sign Language Detection System

> A Vietnamese Sign Language recognition system optimized for CPU and small datasets.

### Quick data pipeline
- Put videos in `Dataset/Video/` and labels in `Dataset/Text/Label.csv` with columns `Video,Label`.
- Run processing: `python main.py --process-data` to generate `Data/keypoints.npy` and `Data/labels.npy`.
- Train: `python main.py --train` to train models and write results to `Models/`.

### Environment requirements
- Python 3.10 (recommended)
- TensorFlow 2.15, MediaPipe 0.10.5, NumPy 1.26.x

### Troubleshooting
- Missing libraries: run `pip install -r requirements.txt` or follow console logs for suggested packages.
- Cannot open camera: check `camera_index` in config or try indices 0/1/2.
- Low FPS: reduce `frame_width/height`, increase `frame_skip` in config.

### Example YAML config (`Configs/config.yaml`)
```yaml
logging:
  level: INFO
  file: Logs/app.log
training:
  camera_index: 0
  frame_width: 640
  frame_height: 480
  fps: 30
  flip_horizontal: true
  frame_skip: 2
  sequence_length: 30
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  prediction_threshold: 0.7
```

## 🚀 Features
- Lightweight Deep Learning models optimized for CPU  
- Real-time video processing (15–25 FPS)  
- Modern and intuitive PyQt5 interface with vertical sidebar  
- Smart data augmentation for limited datasets  
- Supports traditional ML, Deep Learning, and Ensemble methods  
- Modular architecture, detailed logging, and YAML configuration  

## 📁 Main Components
- **Core**: Detection and training engines  
- **Data**: Video processing, keypoints extraction, augmentation  
- **UI**: User interface for real-time visualization  
- **Utils**: Config manager, logging system  
- **Configs**: YAML configuration files  
- **Models**: Trained models storage  
- **Dataset**: Raw video data and labels  
- **Logs**: Application and training logs  

## 🛠️ Installation
Requirements: Python 3.8+, ≥8GB RAM, camera for real-time detection  

```bash
pip install -r requirements.txt
# or
pip install tensorflow-cpu opencv-python mediapipe PyQt5 scikit-learn
```

## 🚀 Usage
- Run GUI application:  
  `python main.py --gui`  

- Train model:  
  `python main.py --train --config custom.yaml`  

- Process video dataset:  
  `python main.py --process-data`  

## 📈 Demo Workflow
1. Prepare Dataset  
   - Place videos in `Dataset/Video/`  
   - Store labels in `Dataset/Text/Label.csv`  

2. Process Data  
   - Run `python main.py --process-data`  
   - Extract keypoints, augment data, save to `Data/`  

3. Train Model  
   - Run `python main.py --train`  
   - Perform cross-validation, select best model, save to `Models/`  

4. Run Real-Time Detection  
   - Run `python main.py --gui`  
   - Load trained model, open camera, recognize signs in real-time  
