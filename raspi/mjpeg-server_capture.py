#!/usr/bin/python3

import io
import os
import time
import json
import logging
import socketserver
import cv2
from http import server
from threading import Condition, Thread
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
        fetch("/capture?label=" + label)
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
        fetch(`/set_setting?setting=${{setting}}&value=${{value}}`);
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
    <button onclick="captureImage()">Capture</button>
    <p>Shortcut Key: Press <b>'c'</b> to capture</p>

    <h2>Camera Settings</h2>
    <label>Gain:</label>
    <input type="range" min="1" max="16" step="0.1" value="1.0" oninput="adjustSetting('gain', this.value)">
    <br>
    <label>Brightness:</label>
    <input type="range" min="0.0" max="1.0" step="0.1" value="0.5" oninput="adjustSetting('brightness', this.value)">
    <br>
    <label>Contrast:</label>
    <input type="range" min="0.5" max="2.0" step="0.1" value="1.0" oninput="adjustSetting('contrast', this.value)">
    <br>
    <label>White Balance:</label>
    <select onchange="adjustSetting('awb_mode', this.value)">
        <option value="auto">Auto</option>
        <option value="tungsten">Tungsten</option>
        <option value="fluorescent">Fluorescent</option>
        <option value="daylight">Daylight</option>
        <option value="cloudy">Cloudy</option>
    </select>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
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
        """Handles image capture and saves it to the dataset."""
        try:
            label = self.path.split('label=')[-1]
            if label not in labels:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid label')
                return
            
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            timestamp = int(time.time())
            filename = os.path.join(label_dir, f"{label}_{timestamp}.jpg")

            frame = picam2.capture_array()
            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imwrite(filename, resized_frame)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"message": "Image saved"}')

        except Exception as e:
            logging.error(f"Error capturing image: {str(e)}")

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
                if setting in ["gain", "brightness", "contrast"]:
                    picam2.set_controls({setting.capitalize(): float(value)})
                elif setting == "awb_mode":
                    picam2.set_controls({"AwbMode": value})
            
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

# Initialize Camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Server started at http://<raspberry_pi_ip>:8000")
    server.serve_forever()
finally:
    picam2.stop_recording()

