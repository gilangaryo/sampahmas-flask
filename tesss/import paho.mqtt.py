import paho.mqtt.client as mqtt
import json
import base64
import uuid
import os
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Konfigurasi MQTT
MQTT_BROKER = "ge9c6717.ala.asia-southeast1.emqxsl.com"
MQTT_PORT = 8883
MQTT_TOPIC = "vending_machine/image"
MQTT_USERNAME = "sampahmas"
MQTT_PASSWORD = "sampahmas123"

# Inisialisasi OpenCV DNN Model untuk deteksi
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_bottle(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "bottle":
                logging.info("Bottle detected!")
                return True
    logging.info("No bottle detected.")
    return False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to MQTT broker!")
        client.subscribe(MQTT_TOPIC)
        logging.info(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        logging.error(f"Failed to connect, return code {rc}")

        
def on_message(client, userdata, msg):
    try:
        # Parse the JSON message payload
        data = json.loads(msg.payload.decode('utf-8'))
        vending_machine_id = data.get("vending_machine_id")
        image_base64 = data.get("image")

        if not vending_machine_id or not image_base64:
            logging.error("Invalid message format or missing fields")
            return

        # Decode the image from Base64
        try:
            image_data = base64.b64decode(image_base64)
        except base64.binascii.Error as e:
            logging.error(f"Failed to decode Base64 image: {e}")
            return

        # Generate a unique filename using UUID
        unique_file_name = f"{uuid.uuid4()}_{vending_machine_id}.jpg"

        # Ensure that the "tmp" directory exists
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

        # Save the decoded image to a file
        file_path = os.path.join("tmp", unique_file_name)
        try:
            with open(file_path, "wb") as image_file:
                image_file.write(image_data)
            logging.info(f"Image received and saved as {file_path}")
        except IOError as e:
            logging.error(f"Failed to save image to file: {e}")
            return

        # Read the image file for processing
        frame = cv2.imread(file_path)
        if frame is not None:
            # Detect the bottle in the image
            if detect_bottle(frame):
                logging.info("Bottle detected from MQTT image!")
            else:
                logging.info("No bottle detected from MQTT image!")
        else:
            logging.error("Failed to read the image file after saving.")

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON message: {e}")
    except Exception as e:
        logging.error(f"Failed to process message: {e}")

# Buat client MQTT
client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

# Tambahkan pengaturan TLS jika menggunakan koneksi SSL
client.tls_set()  # Gunakan sertifikat root default dari OS

# Connect to the broker
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()  # Jalankan loop di background

# Loop utama Flask untuk tetap aktif
try:
    while True:
        pass  # Ini akan menjaga loop MQTT tetap berjalan di background

except KeyboardInterrupt:
    client.loop_stop()
    logging.info("MQTT client stopped")
