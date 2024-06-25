from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('emotion_detection_model_acc_67.51.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('./faceidentifier/haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    emotions = []
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) of the face
        roi_gray = gray[y:y + h, x:x + w]
        resized = cv2.resize(roi_gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        
        # Predict emotion
        prediction = model.predict(reshaped)
        emotion_label_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_label_index]
        
        emotions.append({
            'emotion': emotion_label,
            'bounding_box': [int(x), int(y), int(w), int(h)]
        })
    
    return emotions

@app.route('/detect_emotion_image', methods=['POST'])
def detect_emotion_image():
    data = request.json
    image_data = base64.b64decode(data['image'])
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    emotions = detect_emotion(image)
    
    return jsonify(emotions)

@app.route('/detect_emotion_video', methods=['POST'])
def detect_emotion_video():
    data = request.json
    frames_data = data['frames']
    all_emotions = []
    
    for frame_data in frames_data:
        image_data = base64.b64decode(frame_data)
        np_image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        emotions = detect_emotion(image)
        all_emotions.append(emotions)
    
    return jsonify(all_emotions)

if __name__ == '__main__':
    app.run(debug=True)
