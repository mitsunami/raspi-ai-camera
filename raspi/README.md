# Setup

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all
sudo apt install imx500-tools
sudo apt install python3-picamera2 --no-install-recommends
```

```bash
sudo reboot
```

```bash
sudo dmesg | grep imx500
```


```bash
rpicam-vid -t 10s -o output.h264 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --width 1920 --height 1080 --framerate 30
```


# Preview AI Camera Stream

```bash
git clone https://github.com/raspberrypi/picamera2.git
cd picamera2/examples
python mjpeg-server.py
# Access to <raspberry pi ip address>:8000 (on host web browser)
```


# Data Collection

Create a file called labels.txt with one label per line:
```
rock
paper
scissors
```


```bash
python mjpeg-server_capture.py
#Access to <raspberry pi ip address>:8000 (on host web browser)

# Copy generated dataset to host machine
scp -r pi@<raspberry pi ip address>:/home/pi/raspi-ai-camera/custom_dataset .
```
