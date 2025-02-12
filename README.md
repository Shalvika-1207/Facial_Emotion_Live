# Emotion Detection System

This project is a real-time emotion detection system that identifies emotions from a person's facial expression using a webcam. The system uses a Convolutional Neural Network (CNN) model, trained on the FER2013 dataset, and OpenCV for facial detection.

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- imutils
- Numpy
- pandas

You can install the necessary libraries using `pip`:

```bash
pip install tensorflow opencv-python imutils numpy pandas
```

## Files Description

- **haarcascade_frontalface_default.xml**: Pre-trained model for detecting faces in an image.
- **_mini_XCEPTION.02-0.53.hdf5**: Trained emotion classification model.
- **fer2013.csv**: The dataset used for training the emotion classifier.
- **emojis/**: Folder containing emoji images corresponding to each emotion.
- **emotion_detector.py**: Main Python script that runs the webcam emotion detection.

## How to Use

1. Download the project and place the required models (`haarcascade_frontalface_default.xml` and `_mini_XCEPTION.02-0.53.hdf5`) in the appropriate folders.
2. Open a terminal, navigate to the project directory, and run the `emotion_detector.py` script:

```bash
python emotion_detector.py
```

3. The webcam will open, and as it detects a face, it will display the predicted emotion and the corresponding probability.
4. Press 'q' to exit the webcam stream.

## Model Details

- **Face Detection**: Uses the OpenCV Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect faces.
- **Emotion Classification**: A CNN model trained on the FER2013 dataset to classify emotions into one of the following categories:
  - Angry
  - Disgust
  - Scared
  - Happy
  - Sad
  - Surprised
  - Neutral

## Training

To train the emotion classifier, the following steps were followed:

1. **Data Preprocessing**: The FER2013 dataset was processed to extract pixel values and reshape them for model training.
2. **Model Architecture**: A custom `mini_XCEPTION` model was built using Keras with layers like Convolution2D, BatchNormalization, SeparableConv2D, and MaxPooling2D.
3. **Training**: The model was trained for 20 epochs using the `Adam` optimizer and `categorical_crossentropy` loss.

## Future Improvements

- Implement emotion recognition on videos with multiple faces.
- Add features for facial expression visualization (like emoji reactions).
- Optimize the model for faster inference.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER2013 dataset for emotion classification.
- OpenCV for face detection.
- Keras/TensorFlow for model development.
```

You can adapt this based on additional details you want to include. Let me know if you'd like any changes!
