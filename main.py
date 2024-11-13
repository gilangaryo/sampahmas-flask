from flask import Flask, request, jsonify
import cv2
import cvlib as cv
import numpy as np
import requests
import logging
import firebase_admin
from firebase_admin import credentials, storage
import uuid
import os
from werkzeug.utils import secure_filename

import threading
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


# production
service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS').strip()

if service_account_path is None:
    raise ValueError("Environment variable GOOGLE_APPLICATION_CREDENTIALS not set")






cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sampahmas-3a4f0.appspot.com'
})

bucket = storage.bucket()

def detect_bottle_and_draw(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    bottle_found = False
    percentage = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "bottle":
                bottle_found = True
                percentage = int(confidence * 100)
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{CLASSES[idx]}: {percentage}%"
                y = startY - 8 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return bottle_found, frame, percentage

def upload_to_firebase(file_path, file_name):
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_path)
    blob.make_public()
    logging.info(f"File uploaded to Firebase Storage: {blob.public_url}")
    return blob.public_url

def send_data_to_node_api(url):
    NODE_API_URL = 'https://sampahmas-backend-ox9f.onrender.com/api/endpoint'
    data = {"message": "Bottle detected", "url": url}

    try:
        response = requests.post(NODE_API_URL, json=data)
        if response.status_code == 200:
            logging.info("Data sent to Node.js API successfully")
        else:
            logging.error(f"Failed to send data to Node.js API: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to Node.js API: {e}")

def background_task(original_path, annotated_path, original_name, annotated_name, percentage):
    try:
        original_blob = bucket.blob(f'vending/original/{original_name}')
        original_blob.upload_from_filename(original_path)
        original_blob.make_public()
        original_url = original_blob.public_url
        logging.info(f"Uploaded original image: {original_url}")

        annotated_blob = bucket.blob(f'vending/label/{percentage}_{annotated_name}')
        annotated_blob.upload_from_filename(annotated_path)
        annotated_blob.make_public()
        annotated_url = annotated_blob.public_url
        logging.info(f"Uploaded annotated image with percentage: {annotated_url}")

        return original_url, annotated_url
    except Exception as e:
        logging.error(f"Error in background task: {str(e)}")
        return None, None
@app.route('/')
def home():
    return '''
    <h1>Image Recognition API</h1>
    <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" name="imageFile" accept="image/png, image/jpeg">
        <input type="submit" value="Upload">
    </form>
    '''


def background_task(file_path, unique_file_name):
    try:
        firebase_url = upload_to_firebase(file_path, unique_file_name)
        send_data_to_node_api(firebase_url)
    except Exception as e:
        logging.error(f"Error in background task: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'imageFile' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['imageFile']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_bytes = file.read()
        frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400

        if not os.path.exists("tmp"):
            os.makedirs("tmp")

        filename = secure_filename(file.filename)
        original_file_path = os.path.join("tmp", filename)
        with open(original_file_path, 'wb') as f:
            f.write(file_bytes)

        bottle_found, annotated_frame, percentage = detect_bottle_and_draw(frame)
        annotated_file_name = f"annotated_{uuid.uuid4()}_{filename}"
        annotated_file_path = os.path.join("tmp", annotated_file_name)
        cv2.imwrite(annotated_file_path, annotated_frame)

        if bottle_found:
            unique_file_name = f"{uuid.uuid4()}_{filename}"
            executor = ThreadPoolExecutor(max_workers=3)
            executor.submit(background_task, original_file_path, annotated_file_path, unique_file_name, annotated_file_name, percentage)
            executor.shutdown(wait=False)

            return jsonify({"message": "Bottle detected", "status": True, "confidence": percentage}), 200
        else:
            return jsonify({"message": "No bottle detected", "status": False}), 200
    else:
        return jsonify({"error": "Invalid file format. Only PNG and JPEG are accepted."}), 400

if __name__ == '__main__':
    app.run()
