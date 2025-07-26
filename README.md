# Gesture Classification Using Custom Dataset

A lightweight Temporal Convolutional Network (TCN) for real-time hand gesture recognition using custom wearable sensor data.

## Overview

This project implements a compact TCN architecture for classifying 12 hand gestures from raw sensor data collected using a custom MotionX wearable device. The model achieves **98.02%** accuracy while maintaining computational efficiency suitable for embedded applications.

## Features

- **Lightweight Architecture**: Only **5,804 parameters**
- **High Accuracy**: **98.02%** test accuracy
- **Real-time Performance**: Sub-3ms inference time
- **Embedded Ready**: Optimized for ESP32 deployment (**36.8 KB** model size)
- **Multi-modal Sensing**: Combines Hall-effect sensors and IMU data

## Gesture Classes

The system recognizes **12** distinct hand gestures:

1. Clap
2. Clockwise
3. CounterClockwise
4. Down
5. Idle
6. LeftClick
7. LeftSwipe
8. RightClick
9. RightSwipe
10. Throw
11. Up
12. Wave

## Dataset

- **Total Samples**: 4,386 gesture recordings
- **Users**: 5 participants
- **Sampling Rate**: 100 Hz
- **Duration**: 15 seconds per trial (1,500 samples)
- **Sensors**: 3 Hall-effect sensors + 6-axis IMU

## Model Architecture

- **Type**: Temporal Convolutional Network (TCN)
- **Input**: 9-dimensional feature vectors over 10-step sequences
- **Architecture**: 3 cascaded residual blocks with dilated convolutions
- **Optimization**: Adam optimizer with early stopping
- **Deployment**: TensorFlow Lite with INT8 quantization

## Requirements

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

## Usage

### Data Preprocessing

```bash
# Convert JSON files to CSV
python preprocess.py --input data/json/ --output data/csv/

# Apply filtering and normalization
python preprocess.py --normalize --filter --input data/csv/ --output data/processed/

# Split into train/validation/test sets
python split_data.py --input data/processed/ --train 0.8 --val 0.1 --test 0.1
```

### Model Training

```bash
# Train TCN model
python train_tcn.py --data data/processed/ --epochs 100 --batch_size 32 --patience 10

# Save trained model
python train_tcn.py --save_model model/tcn_model.h5
```

### Model Deployment

```bash
# Convert to TensorFlow Lite
python convert_tflite.py --input model/tcn_model.h5 --output model/tcn_model.tflite

# Apply INT8 quantization
python quantize_model.py --input model/tcn_model.tflite --output model/tcn_model_int8.tflite

# Deploy to ESP32
# (Use ESP32 upload tools to flash the model and inference code)
``` 

## Results

- **Test Accuracy**: 98.02%
- **Model Size**: 36.8 KB (quantized)
- **Inference Time**: 2.5 ms
- **Memory Usage**: 48 KB
- **Energy per Inference**: 15.2 mJ

## Hardware

- **Target Platform**: ESP32 microcontroller
- **Custom Device**: MotionX wearable
- **Sensors**: 3x Hall-effect + 6-axis IMU
- **Sampling**: 100 Hz

## Authors

- **Kiman Adhikari** - Department of Electronics and Computer Engineering, Thapathali Campus, Kathmandu, Nepal
- **Sagar Joshi** - Department of Electronics and Computer Engineering, Thapathali Campus, Kathmandu, Nepal

## Acknowledgements

Special thanks to Rabin Rai and Sanjeep Kumar Sharma for their invaluable contributions and support.

