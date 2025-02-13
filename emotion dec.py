import cv2
from keras.models import load_model
import numpy as np

# Load the model (verify path and model integrity)
model = load_model('C:\\Users\\Samanth Abbur\\Desktop\\cnn\\model.h5')

# Start video capture
cap = cv2.VideoCapture(0)

# Define emotions list
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    ret, frame = cap.read()

    # Face detection (adjust parameters or try alternate detectors if needed)
    face_cascade = cv2.CascadeClassifier('C:\\Users\\Samanth Abbur\\Desktop\\cnn\\haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)  # Adjusted parameters

    for (x, y, w, h) in faces:
        # Preprocess the detected face (ensure consistency with training)
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0  # Ensure float32 type

        # Make prediction (and print raw scores for inspection)
        prediction = model.predict(np.expand_dims(roi_normalized, axis=0))
        print(prediction)  # Print raw scores
        predicted_emotion = emotions[np.argmax(prediction)]

        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
