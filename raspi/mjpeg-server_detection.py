#!/usr/bin/python3

import io
import os
import sys
import time
import json
import logging
import socketserver
import libcamera
from functools import lru_cache
from http import server
from threading import Condition
import argparse
import numpy as np
import cv2
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from picamera2.devices import IMX500
from picamera2.devices.imx500.postprocess import softmax
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
# Global Variables
last_detections = []
LABELS = None

# Set up MJPEG Streaming Page (Web Interface)
PAGE = """\
<html>
<head>
<title>Raspberry Pi Detection Stream</title>
</head>
<body>
    <h1>Raspberry Pi Detection Stream</h1>
    <img src="stream.mjpg" width="640" height="480" /><br>
    <h2>Live Detection Results</h2>
    <div id="results">Waiting for detection...</div>

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
                .catch(error => console.error("Error fetching detection:", error));
        }
        setInterval(fetchResults, 1000);  // Refresh every second
        function adjustSetting(setting, value) {
            fetch(`/set_setting?setting=${setting}&value=${value}`);
        }
    </script>
</body>
</html>
"""

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_detections
    if detections is None:
        return
    #labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{intrinsics.labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
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
    """Handles HTTP requests for MJPEG streaming and detection results."""
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
        """Handles the request to fetch detection results."""
        global last_detections
        results_text = "<br>".join([
            f"{intrinsics.labels[int(d.category)]}: {d.conf:.2f}"
            for d in last_detections
        ])
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
    intrinsics.task = "object detection"

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    # Initialize Camera and Stream
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    picam2.configure(config)

    imx500.show_network_fw_progress_bar()
    
    # Start MJPEG Stream
    output = StreamingOutput()
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))
    picam2.pre_callback = draw_detections

    # Start HTTP Server
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        print("Server started at http://<raspberry_pi_ip>:8000")
        server.serve_forever()
    finally:
        picam2.stop()


