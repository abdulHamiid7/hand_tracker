# hand_tracker


Real-time hand tracking, distance measurement and gesture recognition.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python hand_tracker.py                 # default webcam
    python hand_tracker.py --camera 1      # another camera
    python hand_tracker.py --calib-dist 50 # calibration distance in cm

Keys:
    c  - calibrate camera distance (hold your open hand at --calib-dist cm from the camera)
    m  - mirror view on/off
    q  - quit

What it measures:
    * Pinch distance : thumb tip <-> index tip, in cm (scaled using your palm width)
    * Camera distance: hand -> camera, in cm (pinhole model using palm width in pixels)
    * Gesture        : Thumbs Up/Down, Fist, Open Palm, Pointing, Peace, OK, Rock, Three, Four
