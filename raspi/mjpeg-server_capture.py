#!/usr/bin/python3

import io
import os
import time
import json
import logging
import socketserver
import libcamera
import threading
import cv2
from http import server
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# Load labels from labels.txt
labels_file = "labels.txt"
with open(labels_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create dataset directories
dataset_dir = "./custom_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# MJPEG Streaming Page (Updated with capture button & controls)
PAGE = """\
<html>
<head>
<title>Raspberry Pi Camera Stream</title>
<script>
    function captureImage() {
        let label = document.getElementById("label").value;
        let mode = document.getElementById("mode").value;
        let numPhotos = document.getElementById("numPhotos").value;
        let interval = document.getElementById("interval").value;
        let sharpness = document.getElementById("sharpness").value;
        let url = `/capture?label=${label}&mode=${mode}&numPhotos=${numPhotos}&interval=${interval}&sharpness=${sharpness}`;
        fetch(url)
            .then(() => updateCounts())  // Update count immediately after capture
            .catch(error => console.error("Error capturing image:", error));
    }

    function updateCounts() {
        fetch("/label_counts")
            .then(response => response.json())
            .then(data => {
                let dropdown = document.getElementById("label");
                for (let i = 0; i < dropdown.options.length; i++) {
                    let label = dropdown.options[i].value;
                    if (data[label] !== undefined) {
                        dropdown.options[i].text = label + " (" + data[label] + " images)";
                    }
                }
            })
            .catch(error => console.error("Error fetching label counts:", error));
    }

    setInterval(updateCounts, 5000);  // Refresh counts every 5 seconds

    function adjustSetting(setting, value) {
        fetch(`/set_setting?setting=${setting}&value=${value}`);
    }

    document.addEventListener("keydown", function(event) {
        if (event.key === "c") {
            captureImage();
        }
    });
</script>
</head>
<body onload="updateCounts()">
    <h1>Raspberry Pi Camera Stream</h1>
    <img src="stream.mjpg" width="640" height="480" /><br>

    <h2>Capture Image</h2>
    <label for="label">Label:</label>
    <select id="label">
        {options_placeholder}
    </select>
    <br>
    <br>

    <div id="autoSettings">
        <h3>Capture Settings</h3>
        <label for="mode">Capture Mode:</label>
        <select id="mode">
            <option value="manual">Manual Capture</option>
            <option value="auto">Automatic Capture</option>
        </select>
        <br>
 
        <label for="numPhotos">Number of Photos:</label>
        <input type="number" id="numPhotos" min="1" max="100" value="10">
        <br>

        <label for="interval">Interval (seconds):</label>
        <input type="number" id="interval" min="0.1" max="10" step="0.1" value="0.5">
        <br>

        <label for="sharpness">Sharpness Threshold:</label>
        <input type="number" id="sharpness" min="30" max="200" value="50">
        <br>
    </div>

    <button onclick="captureImage()">Capture</button>
    <p>Shortcut Key: Press <b>'c'</b> to start capturing!<br>
        <b>Manual Capture:</b> Takes a single image instantly.<br>
        <b>Automatic Capture:</b> Captures multiple images at regular intervals & filters out blurry ones. <br>
    </p>

    <h2>Camera Settings</h2>
    <label>Analogue Gain:</label>
    <input type="range" min="1" max="16" step="0.1" value="6.0" oninput="adjustSetting('gain', this.value)">
    <br>
    <label>Brightness:</label>
    <input type="range" min="-1.0" max="1.0" step="0.1" value="0.0" oninput="adjustSetting('brightness', this.value)">
    <br>
    <label>Contrast:</label>
    <input type="range" min="0.0" max="32.0" step="0.1" value="1.0" oninput="adjustSetting('contrast', this.value)">
    <br>
    <label>Exposure:</label>
    <select onchange="adjustSetting('ae_mode', this.value)">
        <option value="normal">Normal</option>
        <option value="short">Short</option>
        <option value="long">Long</option>
    </select>
    <br>
    <label>Exposure Value:</label>
    <input type="range" min="-8.0" max="8.0" step="0.33" value="0.0" oninput="adjustSetting('exposure', this.value)">
    <br>
    <label>White Balance:</label>
    <select onchange="adjustSetting('awb_mode', this.value)">
        <option value="auto">Auto</option>
        <option value="tungsten">Tungsten</option>
        <option value="fluorescent">Fluorescent</option>
        <option value="indoor">Indoor</option>
        <option value="daylight">Daylight</option>
        <option value="cloudy">Cloudy</option>
    </select>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        global capture_mode, num_photos, capture_interval, sharpness_threshold

        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            #content = PAGE.encode('utf-8')
            label_counts = get_label_counts()
            options = ''.join([f'<option value="{label}">{label} ({label_counts[label]} images)</option>' for label in labels])
            content = PAGE.replace("{options_placeholder}", options).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        elif self.path.startswith('/capture'):
            self.handle_capture()
        elif self.path.startswith('/label_counts'):
            self.handle_label_counts()
        elif self.path.startswith('/set_setting'):
            self.handle_set_setting()
        else:
            self.send_error(404)
            self.end_headers()

    def handle_capture(self):
        """Handles manual or automatic image capture based on selected mode."""
        global capture_active
        params = dict(item.split('=') for item in self.path.split('?')[-1].split('&'))

        label = params.get("label")
        mode = params.get("mode")
        num_photos = int(params.get("numPhotos", 10))
        capture_interval = float(params.get("interval", 2))
        sharpness_threshold = int(params.get("sharpness", 100))

        if mode == "manual":
            capture_manual_photo(label)
        elif mode == "auto":
            capture_active = True
            threading.Thread(target=auto_capture_photos, args=(label, num_photos, capture_interval, sharpness_threshold)).start()

        self.send_response(200)
        self.end_headers()

    def handle_label_counts(self):
        """Returns the number of images in each label directory as JSON."""
        label_counts = get_label_counts()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(label_counts).encode('utf-8'))

    def handle_set_setting(self):
        """Adjusts camera settings dynamically."""
        try:
            query = self.path.split('?')[-1]
            params = dict(item.split('=') for item in query.split('&'))
            setting = params.get("setting")
            value = params.get("value")

            if setting and value:
                if setting in ["gain"]:
                    picam2.set_controls({"AnalogueGain": float(value)})
                elif setting in ["exposure"]:
                    picam2.set_controls({"ExposureValue": float(value)})
                elif setting in ["brightness", "contrast", "saturation", "sharpness"]:
                    picam2.set_controls({setting.capitalize(): float(value)})
                elif setting == "ae_mode":
                    if value == "normal":
                        mode = libcamera.controls.AeExposureModeEnum.Normal
                    elif value == "short":
                        mode = libcamera.controls.AeExposureModeEnum.Short
                    elif value == "long":
                        mode = libcamera.controls.AeExposureModeEnum.Long
                    picam2.set_controls({"AeExposureMode": mode})
                elif setting == "awb_mode":
                    if value == "auto":
                        mode = libcamera.controls.AwbModeEnum.Auto
                    elif value == "tungsten":
                        mode = libcamera.controls.AwbModeEnum.Tungsten
                    elif value == "fluorescent":
                        mode = libcamera.controls.AwbModeEnum.Fluorescent
                    elif value == "indoor":
                        mode = libcamera.controls.AwbModeEnum.Indoor
                    elif value == "daylight":
                        mode = libcamera.controls.AwbModeEnum.Daylight
                    elif value == "cloudy":
                        mode = libcamera.controls.AwbModeEnum.Cloudy
                    picam2.set_controls({"AwbMode": mode})
                else:
                    logging.warning(f"Unknown setting: {setting}")
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"message": "Setting updated"}')

        except Exception as e:
            logging.error(f"Error setting camera parameters: {str(e)}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Failed to update setting"}')

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def get_label_counts():
    """Returns a dictionary with label names as keys and image counts as values."""
    label_counts = {}
    for label in labels:
        label_dir = os.path.join(dataset_dir, label)
        if os.path.exists(label_dir):
            label_counts[label] = len([f for f in os.listdir(label_dir) if f.endswith('.jpg')])
        else:
            label_counts[label] = 0
    return label_counts

def is_sharp(image, threshold=100):
    """Determines if an image is sharp using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"sharpness: {laplacian_var}, threshold: {threshold}")
    return laplacian_var > threshold

def capture_manual_photo(label):
    """Captures a single image manually and saves it."""
    label_dir = os.path.join(dataset_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    frame = picam2.capture_array()
    filename = os.path.join(label_dir, f"{label}_{int(time.time()*1000)}.jpg")
    request = picam2.capture_request()
    request.save("main", filename)
    request.release()

    #cv2.imwrite(filename, frame)
    print(f"Manual Capture Saved: {filename}")

def auto_capture_photos(label, num_photos, interval, sharpness_threshold=100):
    """Automatically captures multiple images but only saves sharp ones."""
    global capture_active
    label_dir = os.path.join(dataset_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    saved_images = 0
    while saved_images < num_photos and capture_active:
        frame = picam2.capture_array()
        if is_sharp(frame, threshold=sharpness_threshold):
            filename = os.path.join(label_dir, f"{label}_{int(time.time()*1000)}.jpg")
            request = picam2.capture_request()
            request.save("main", filename)
            request.release()
            #cv2.imwrite(filename, frame)
            saved_images += 1
            print(f"Saved: {filename} (Sharp Image)")
        else:
            print("Skipped a blurry image...")

        time.sleep(interval)

    capture_active = False


# Initialize Camera
picam2 = Picamera2()
#picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.set_controls({"AeEnable": 1})
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Server started at http://<raspberry_pi_ip>:8000")
    server.serve_forever()
finally:
    picam2.stop_recording()

