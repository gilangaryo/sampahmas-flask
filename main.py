from flask import Flask, request, jsonify
import cv2
import cvlib as cv
# from cvlib.object_detection import draw_bbox  # Uncomment if you want to draw bounding boxes
import numpy as np
import requests
import os
from werkzeug.serving import WSGIRequestHandler

# Disable TensorFlow AVX warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Allow up to 16 MB

WSGIRequestHandler.timeout = 60 
def detect_bottle(frame):
    # Use cvlib to detect common objects in the image
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

    # Optional: Draw bounding boxes and save the image
    # processed_frame = draw_bbox(frame, bbox, label, conf)
    # cv2.imwrite('processed_image.jpg', processed_frame)

    # Check if 'bottle' is among the detected objects
    if 'bottle' in label:
        print("Bottle detected!")
        return True
    else:
        print("No bottle detected.")
        return False

def send_data_to_node_api():
    # Replace with your Node.js API URL
    NODE_API_URL = 'https://m2bvdfxc-3000.asse.devtunnels.ms/api/endpoint'

    data = {
        "message": "Bottle detected",
        # Add additional data if needed
    }

    try:
        response = requests.post(NODE_API_URL, json=data)
        if response.status_code == 200:
            print("Data sent to Node.js API successfully")
        else:
            print(f"Failed to send data to Node.js API: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to Node.js API: {e}")

# get homepage

@app.route('/')
def home():
    return '''
    <h1>Image Recognition API</h1>
    <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" name="imageFile">
        <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if an image file is part of the request
    if 'imageFile' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['imageFile']
    print("Uploading")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Convert the uploaded file to a NumPy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        # Decode the image to a format OpenCV can work with
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect bottle in the image
        bottle_found = detect_bottle(frame)

        if bottle_found:
            # Send data to Node.js API
            # send_data_to_node_api()
            return jsonify({"message": "Bottle detected"}), 200
        else:
            return jsonify({"message": "No bottle detected"}), 200

if __name__ == '__main__':
    app.run()
