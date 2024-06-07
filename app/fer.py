import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask import Blueprint
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import base64

# 앱 등록
fer_app = Blueprint('fer', __name__)

# 모델 로드
fer_model = load_model('./models/face_emotion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 감정 레이블 딕셔너리
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def detect_emotion(file):
    try:
        nparr = np.frombuffer(base64.b64decode(file), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 이미지를 성공적으로 읽었는지 확인
        if image is None:
            raise ValueError("Failed to read the image")

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        emotions = []

        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]  # 그레이스케일 이미지에서 얼굴 부분 추출
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            prediction = fer_model.predict(img_pixels)

            emotion_label = [label_dict[idx] for idx in np.argmax(prediction, axis=1)]

        return emotion_label

    except Exception as e:
        print("Error:", e)
        return None

# API 엔드포인트
@fer_app.route('/model/fer', methods=['POST'])
def detect_emotion_api():
    if request.method == 'POST':
        file = request.form.get('file')
        pk = request.form.get('pk')

        emotions = detect_emotion(file)

        if emotions:
            response_data = {
                "pk": pk,
                "emotions": emotions
            }
        else:
            response_data = {
                "pk": pk,
                "error": "No faces detected"
            }

        return jsonify(response_data)
    return jsonify({"error": "Invalid request method"})
