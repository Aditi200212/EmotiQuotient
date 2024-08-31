from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Updated import for TensorFlow's Keras
from tensorflow.keras.preprocessing.image import img_to_array 
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
classifier = load_model(r'C:\Users\BUSIREDDY ADITI\Desktop\EmotionDetection\emotiondetection\model.h5')

face_classifier = cv2.CascadeClassifier(r'C:\Users\BUSIREDDY ADITI\Desktop\EmotionDetection\emotiondetection\haarcascade_frontalface_default.xml')


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            emotion = detect_emotion(filepath)
            return redirect(url_for('result', emotion=emotion))

@app.route('/result/<emotion>')
def result(emotion):
    return f'The detected emotion is {emotion}'

def detect_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label

    return 'No Faces Detected'

if __name__ == '__main__':
    app.run(debug=True)
