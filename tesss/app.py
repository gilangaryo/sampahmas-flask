import paho.mqtt.client as mqtt
import json
import base64

MQTT_BROKER = "mqtts://ge9c6717.ala.asia-southeast1.emqxsl.com"  # Ganti dengan alamat broker MQTT
MQTT_PORT = 5000
MQTT_TOPIC = "vending_machine/image"
MQTT_USERNAME = "sampahmas" # Optional, set if your broker requires authentication
MQTT_PASSWORD = "sampahmas123" # Optional, set if your broker requires authentication

# Define callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker!")
        client.subscribe(MQTT_TOPIC)  # Subscribe to a wildcard topic
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        # Parse pesan JSON
        data = json.loads(msg.payload)
        vending_machine_id = data.get("vending_machine_id")
        image_base64 = data.get("image")

        # Dekode gambar dari base64
        image_data = base64.b64decode(image_base64)

        # Simpan gambar ke file
        with open(f"{vending_machine_id}_captured.jpg", "wb") as image_file:
            image_file.write(image_data)

        print(f"Image received and saved as {vending_machine_id}_captured.jpg")
    except Exception as e:
        print(f"Failed to process message: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
