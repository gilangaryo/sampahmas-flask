from flask import Flask, request, jsonify
import cv2
import cvlib as cv
import numpy as np
import requests
# import os
import logging
from werkzeug.serving import WSGIRequestHandler

logging.basicConfig(level=logging.INFO)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

WSGIRequestHandler.timeout = 60

def detect_bottle(frame):
    label  = cv.detect_common_objects(frame)
    if 'bottle' in label:
        logging.info("Bottle detected!")
        return True
    else:
        logging.info("No bottle detected.")
        return False

def send_data_to_node_api():
    NODE_API_URL = 'https://m2bvdfxc-3000.asse.devtunnels.ms/api/endpoint'
    data = {"message": "Bottle detected"}

    try:
        response = requests.post(NODE_API_URL, json=data)
        if response.status_code == 200:
            logging.info("Data sent to Node.js API successfully")
        else:
            logging.error(f"Failed to send data to Node.js API: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to Node.js API: {e}")

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
    logging.info("File uploaded")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        bottle_found = detect_bottle(frame)

        if bottle_found:
            return jsonify({"message": "Bottle detected"}), 200
        else:
            return jsonify({"message": "No bottle detected"}), 200
    else:
        return jsonify({"error": "Invalid file format. Only PNG and JPEG are accepted."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  
