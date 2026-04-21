# Drone Violence Detection with InceptionV3 + BiLSTM

This project is a small violence detection pipeline for video streams. It uses InceptionV3 as a CNN feature extractor and a BiLSTM model for temporal classification. The repository includes the training notebook, a real-time inference script for DJI Tello, and the minimal Python dependencies required to run the code.

## Project structure

```text
drone-violence-detection-inceptionv3-bilstm/
├── README.md
├── requirements.txt
├── notebooks/
│   └── cnn_bilstm.ipynb
└── src/
    └── inference_with_drone.py.py
```

## What each file contains

### `README.md`
Project documentation. This file explains the purpose of the repository and gives a quick description of the folders and files.

### `requirements.txt`
List of the main runtime dependencies:
- `djitellopy` for connecting to and reading video from the DJI Tello drone.
- `opencv-python` for frame processing and on-screen visualization.
- `numpy` for array and sequence handling.
- `tensorflow` for loading the trained CNN + BiLSTM model.

### `notebooks/cnn_bilstm.ipynb`
Main experiment and training notebook. It contains the end-to-end workflow for the violence detection model, including data/model steps, evaluation metrics, plots such as confusion matrix and ROC/PR curves, and saving the trained model as `violence_detection_cnn_bilstm.h5`.

### `src/inference_with_drone.py.py`
Real-time inference script for the trained model. It:
- connects to a DJI Tello drone,
- captures live frames,
- extracts frame features with InceptionV3,
- builds a sequence of frames for BiLSTM prediction,
- applies thresholding and temporal smoothing,
- shows predictions on screen and writes logs to an `outputs` folder.

## Summary

If you want to understand the training pipeline, start with `notebooks/cnn_bilstm.ipynb`. If you want to run live detection with the trained model and a Tello drone, use `src/inference_with_drone.py.py`.
