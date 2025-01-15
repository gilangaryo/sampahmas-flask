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
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024

# DEVELOPMENT
# if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\code\backend-sampahmas-deteksi\serviceAccount.json'

# service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
# if service_account_path is None:
#     raise ValueError("Environment variable GOOGLE_APPLICATION_CREDENTIALS not set")

# cred = credentials.Certificate(service_account_path)
# firebase_admin.initialize_app(cred, {
#     'storageBucket': 'sampahmas-3a4f0.appspot.com'
# })

# PRODUCTION
service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS').strip()

if service_account_path is None:
    raise ValueError("Environment variable GOOGLE_APPLICATION_CREDENTIALS not set")

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sampahmas-3a4f0.appspot.com'
})

bucket = storage.bucket()

def detect_bottle_and_draw(frame):
    import time

    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Mulai mencatat waktu total
    total_start_time = time.time()

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Mulai mencatat waktu inferensi
    start_time = time.time()
    detections = net.forward()
    end_time = time.time()

    inference_time = end_time - start_time

    bottle_found = False
    percentage = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "bottle":
                logging.info("Bottle detected!")
                bottle_found = True
                percentage = int(confidence * 100)

                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{CLASSES[idx]}: {percentage}%"
                y = startY - 8 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                break

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logging.info(f"Inference time: {inference_time:.4f} seconds")
    logging.info(f"Total time (preprocessing + inference + postprocessing): {total_time:.4f} seconds")

    return bottle_found, frame, percentage, inference_time, total_time


def upload_to_firebase(file_path, firebase_path):
    try:
        logging.info("Starting upload to Firebase Storage")
        blob = bucket.blob(firebase_path)
        blob.upload_from_filename(file_path)
        blob.make_public()
        logging.info(f"File uploaded to Firebase Storage!")
        return blob.public_url
    except Exception as e:
        logging.error(f"Error uploading to Firebase Storage: {str(e)}")
        return None

def send_data_to_node_api(url):
    NODE_API_URL = 'https://m2bvdfxc-3000.asse.devtunnels.ms/api/endpoint'
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
        original_firebase_path = f'vending/original/{original_name}'
        annotated_firebase_path = f'vending/label/{percentage}_{annotated_name}'

        original_url = upload_to_firebase(original_path, original_firebase_path)
        annotated_url = upload_to_firebase(annotated_path, annotated_firebase_path)

        if original_url:
            send_data_to_node_api(original_url)
        
        logging.info(f"Original and annotated images uploaded. URLs: {original_url}, {annotated_url}")

        if original_url and annotated_url:
            if os.path.exists(original_path):
                os.remove(original_path)
                logging.info(f"Deleted temporary file: {original_path}")
            if os.path.exists(annotated_path):
                os.remove(annotated_path)
                logging.info(f"Deleted temporary file: {annotated_path}")
        
    except Exception as e:
        logging.error(f"Error in background task: {str(e)}")

@app.route('/')
def home():
    return '''
    <h1>Image Recognition API</h1>
    <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" name="imageFile" accept="image/png, image/jpeg">
        <input type="submit" value="Upload">
    </form>
    '''

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

        bottle_found, annotated_frame, percentage, inference_time, total_time = detect_bottle_and_draw(frame)
        annotated_file_name = f"annotated_{uuid.uuid4()}_{filename}"
        annotated_file_path = os.path.join("tmp", annotated_file_name)
        cv2.imwrite(annotated_file_path, annotated_frame)
        logging.info(f"DETEKSI ??")
        
        if bottle_found:
            logging.info(f"FOUND!!!!!!!! ??")
            unique_file_name = f"{uuid.uuid4()}_{filename}"
            executor = ThreadPoolExecutor(max_workers=3)
            executor.submit(background_task, original_file_path, annotated_file_path, unique_file_name, annotated_file_name, percentage)
            executor.shutdown(wait=False)

            return jsonify({
                "message": "Bottle detected",
                "status": True,
                "confidence": percentage,
                "Detection time": f"{inference_time:.4f} seconds",
                "Total time": f"{total_time:.4f} seconds"
            }), 200

        else:
            return jsonify({
                "message": "No bottle detected",
                "status": False,
                "Detection time": f"{inference_time:.4f} seconds",
                "Total time": f"{total_time:.4f} seconds"
            }), 200

    else:
        return jsonify({"error": "Invalid file format. Only PNG and JPEG are accepted."}), 400

if __name__ == '__main__':
    app.run()
