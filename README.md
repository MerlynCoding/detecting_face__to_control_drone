
# Face Detection and Drone Following

Welcome to the Face Detection and Drone Following project! This project utilizes the DJITelloPy library for controlling the Tello drone, OpenCV for image processing, and Haar cascade classifier for face detection. Let's get started:

## Installation

To install the necessary dependencies, follow these steps:

1. Install DJITelloPy library:

```bash
pip install djitellopy
```

2. Install OpenCV:

```bash
pip install opencv-python-headless
```

3. Download the `haarcascade_frontalface_default.xml` file. You can find it in the OpenCV GitHub repository [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

## Usage

Once you have installed the dependencies and obtained the `haarcascade_frontalface_default.xml` file, you are ready to use the face detection and drone following script.

First, ensure that your Tello drone is connected to your computer and turned on.

Next, run the `final_main_code.py` script. This script will:

1. Initialize the Tello drone.
2. Start the video stream from the drone's camera.
3. Use the Haar cascade classifier to detect faces in the video stream.
4. Make the drone fly following the detected face.

```bash
python final_main_code.py
```

Enjoy watching the Tello drone autonomously follow your face!

