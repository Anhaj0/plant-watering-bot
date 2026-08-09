# 🌿 Plant Watering Bot - Plant Detection System

Created by **[Anhaj Uwaisulkarni](https://anhaj0.github.io/)**, this project trains a custom Convolutional Neural Network (CNN) model to detect whether an object is a **plant** or **not a plant** using images. It is part of a larger robotic system designed to autonomously water plants based on real-time camera input.

## 🚀 Features

- 12-phase image training pipeline  
  - RGB and Grayscale modes  
  - Rotation, brightness, and noise augmentation  
- Transfer learning  
  - EfficientNetB0 base  
  - Fine-tuned with all layers unfrozen  
- Quantization  
  - Final model exported in INT8 TFLite format for embedded devices  
- Live detection support  
  - Compatible with webcam or video input  
  - Ready for ESP32-based robotic deployment

## 📁 Folder Structure

```

plant\_detection/
├── model/                  # Saved Keras and TFLite models
├── scripts/                # Model training and utility scripts
├── run.txt                 # Runtime logs (excluded from git)
├── plants/                 # Training images (excluded from git)
│   ├── plant/
│   └── not\_plant/
├── Train.py                # Main training script
├── detect\_live.py          # Webcam detection (optional)
├── README.md               # This file
├── .gitignore

````

> ✅ `plants/` and `run.txt` are excluded from Git tracking.

## 📦 Requirements

Make sure Python 3.9 or above is installed, then run:

```bash
pip install -r requirements.txt
````

### requirements.txt

```txt
tensorflow
numpy
matplotlib
opencv-python
```

## 🧠 Model Training Pipeline

Training is done in 12 phases:

### RGB Image Training

1. Standard (rotation ±20 degrees, zoom, flip, brightness 0.6 to 1.4)
2. Dark Boost (brightness 0.2 to 0.6)
3. Bright Boost (brightness 1.4 to 2.0)
4. Rot90 (rotate ±90 degrees)
5. Rot180 (rotate ±180 degrees)
6. Rot270 (rotate ±270 degrees)

### Grayscale Image Training

7. Standard
8. Dark Boost
9. Bright Boost
10. Rot90
11. Rot180
12. Rot270

### Final Fine-tuning

* Unfreezes all layers and trains again using full dataset for better accuracy.

## 📉 Sample Accuracy

```
RGB Phase 1 Accuracy: 91.3%
RGB Phase 3 Accuracy: 94.5%
Grayscale Phase 6 Accuracy: 94.0%
Final Fine-tuned Accuracy: 91.0%
```

## 🪄 INT8 TFLite Export

After training, the model is exported for deployment on microcontrollers like ESP32-S3.

```bash
model/
├── plant_model.keras
├── plant_model_int8.tflite
```

## 🔧 Run Live Detection (Optional)

To test on webcam:

```bash
python detect_live.py
```

## 🤖 Project Goal

This is part of a robot that:

* Navigates with wheels
* Detects plants visually
* Waters them using a pump and moisture sensor
* Returns to a charging or refill station if battery or water is low

## 🧑‍💻 Author

**Anhaj Uwaisulkarni**

Website: [anhaj0.github.io](https://anhaj0.github.io/)  
GitHub: [github.com/Anhaj0](https://github.com/Anhaj0)
Project for NIBM Robotics Coursework 2025

## 📜 License

This project is for educational use. Contact the author for commercial use or redistribution.
