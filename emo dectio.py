import cv2
from keras.models import load_model
import numpy as np



model = load_model('C:\\Users\\Samanth Abbur\\Desktop\\cnn\\model.h5')


cap = cv2.VideoCapture(0)


emotions = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

while True:
    
    ret, frame = cap.read()

   
    face_cascade = cv2.CascadeClassifier('C:\\Users\\Samanth Abbur\\Desktop\\cnn\\haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (48, 48))  

       
        roi_normalized = roi_resized / 255.0  

      
        prediction = model.predict(np.expand_dims(roi_normalized, axis=0))
        predicted_emotion = emotions[np.argmax(prediction)]

       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

  
    cv2.imshow('Emotion Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
