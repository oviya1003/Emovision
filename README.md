# Emotion Detection Using CNN

![WhatsApp Image 2025-02-13 at 14 03 03](https://github.com/user-attachments/assets/7d9f3ef9-2a77-4ce9-a843-efaad4c14018)

## Overview
This project implements real-time emotion detection using a Convolutional Neural Network (CNN) and OpenCV. The model detects human faces and classifies their emotions into seven categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

------------



## Features
- Uses a pre-trained CNN model for emotion classification
- Detects faces using OpenCV's Haar Cascade Classifier
- Classifies emotions in real-time using a webcam feed
- Displays bounding boxes around detected faces with corresponding emotion labels

------------


## Requirements
Ensure you have the following dependencies installed before running the project:

- Python 3.x
- OpenCV (`cv2`)
- Keras
- TensorFlow
- NumPy
 
You can install them using:
`pip install opencv-python keras tensorflow numpy`

------------



## Usage
1. Ensure you have a working webcam.
2. Place the trained model.h5 file in the project directory.
3. Run the script:
`python main.py`
4. The webcam will start capturing frames and detecting emotions.
5. Press `q` to exit.

------------

## Notes
- Make sure the paths to `haarcascade_frontalface_alt2.xml` and `model.h5` are correctly set in your scripts.
- Adjust parameters in `detectMultiScale()`to improve face detection accuracy if needed.

------------



## Acknowledgments
- OpenCV for face detection
- Keras & TensorFlow for deep learning models
- Haar Cascade Classifier for facial recognition


