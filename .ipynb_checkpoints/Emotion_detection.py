from keras.models import load_model 
from keras.preprocessing.image import img_to_array 
import cv2
import numpy as np
import os


face_classifier = cv2.CascadeClassifier(r'C:\Users\BUSIREDDY ADITI\Desktop\EmotionDetection\.ipynb_checkpoints\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\BUSIREDDY ADITI\Desktop\EmotionDetection\.ipynb_checkpoints\model.h5')


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


image_path = r"C:\Users\BUSIREDDY ADITI\Desktop\EmotionDetection\.ipynb_checkpoints\images\mix2.jpeg"

img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load image: {image_path}")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with the emotion label
    cv2.imshow('Emotion Detector', img)
    cv2.waitKey(0)  # Wait for a key press to close the window

cv2.destroyAllWindows