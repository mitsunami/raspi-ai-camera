#!/usr/bin/python3

import io
import os
import sys
import time
import json
import logging
import socketserver
import libcamera
from http import server
from threading import Condition
import argparse
import numpy as np
import cv2
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

# Global Variables
last_detections = []
LABELS = None

# Set up MJPEG Streaming Page (Web Interface)
PAGE = """\
<html>
<head>
<title>Raspberry Pi Classification Stream</title>
</head>
<body>
    <h1>Raspberry Pi Classification Stream</h1>
    <img src="stream.mjpg" width="640" height="480" /><br>
    <h2>Live Classification Results</h2>
    <div id="results">Waiting for classification...</div>

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

    <script>
        function fetchResults() {
            fetch("/results")
                .then(response => response.json())
                .then(data => {
                    document.getElementById("results").innerHTML = data.results;
                })
                .catch(error => console.error("Error fetching classification:", error));
        }
        setInterval(fetchResults, 1000);  // Refresh every second
        function adjustSetting(setting, value) {
            fetch(`/set_setting?setting=${setting}&value=${value}`);
        }
    </script>
</body>
</html>
"""

class Classification:
    def __init__(self, idx: int, score: float):
        self.idx = idx
        self.score = score

def get_label(request: CompletedRequest, idx: int) -> str:
    """Retrieve the label corresponding to the classification index."""
    global LABELS
    if LABELS is None:
        LABELS = intrinsics.labels
        #assert len(LABELS) in [1000, 1001], "Labels file should contain 1000 or 1001 labels."
        output_tensor_size = imx500.get_output_shapes(request.get_metadata())[0][0]
        if output_tensor_size == 1000:
            LABELS = LABELS[1:]  # Ignore background label if present
    return LABELS[idx]

def parse_classification_results(request: CompletedRequest):
    """Parse the output tensor into classification results above threshold."""
    global last_detections
    np_outputs = imx500.get_outputs(request.get_metadata())
    if np_outputs is None:
        return last_detections
    np_output = np_outputs[0]
    if intrinsics.softmax:
        np_output = softmax(np_output)
    top_indices = np.argpartition(-np_output, 3)[:3]  # Top 3 indices with the highest scores
    top_indices = top_indices[np.argsort(-np_output[top_indices])]  # Sort top 3 indices by their scores
    last_detections = [Classification(index, np_output[index]) for index in top_indices]
    return last_detections

def draw_classification_results(request: CompletedRequest, results, stream="main"):
    """Draw classification results on the video stream."""
    with MappedArray(request, stream) as m:
        text_x, text_y = 10, 25  # Position for text
        for index, result in enumerate(results):
            label = get_label(request, idx=result.idx)
            #text = f"{label}: {result.score:.3f}"
            text = f"{label}"
            if result.score > 0.7:
                cv2.putText(m.array, text, (text_x, text_y + index * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            break

def parse_and_draw_classification_results(request: CompletedRequest):
    """Analyse and draw the classification results in the output tensor."""
    results = parse_classification_results(request)
    draw_classification_results(request, results)

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk")
    parser.add_argument("--labels", type=str, help="Path of labels file")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("-s", "--softmax", action=argparse.BooleanOptionalAction, help="Apply softmax")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction, help="Keep aspect ratio")
    return parser.parse_args()

class StreamingOutput(io.BufferedIOBase):
    """Handles MJPEG streaming output."""
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    """Handles HTTP requests for MJPEG streaming and classification results."""
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
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
                logging.warning('Streaming client removed: %s', str(e))
        elif self.path == '/results':
            self.handle_results()
        elif self.path.startswith('/set_setting'):
            self.handle_set_setting()
        else:
            self.send_error(404)
            self.end_headers()

    def handle_results(self):
        """Handles the request to fetch classification results."""
        global last_detections
        results_text = "<br>".join([f"{get_label(None, d.idx)}: {d.score:.3f}" for d in last_detections])
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"results": results_text}).encode('utf-8'))

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
    """Handles streaming server requests."""
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    args = get_args()

    # Initialize IMX500 model
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "classification"

    # Load labels if provided
    if args.labels:
        with open(args.labels, 'r') as f:
            intrinsics.labels = f.read().splitlines()
    else:
        with open("assets/imagenet_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    # Initialize Camera and Stream
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    picam2.configure(config)

    imx500.show_network_fw_progress_bar()
    
    # Start MJPEG Stream
    output = StreamingOutput()
    #picam2.start(config, show_preview=True)
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))
    picam2.pre_callback = parse_and_draw_classification_results

    # Start HTTP Server
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        print("Server started at http://<raspberry_pi_ip>:8000")
        server.serve_forever()
    finally:
        picam2.stop()

