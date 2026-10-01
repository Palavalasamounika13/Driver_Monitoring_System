# DRIVER MONITORING SYSTEM - Streamlit Cloud version

import os
import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer

# CONFIGURATION

# model file must sit in the repo next to app.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

EAR_THRESHOLD = 0.20
BLINK_THRESHOLD = 0.70
MAR_THRESHOLD = 0.60
DROWSY_TIME = 1.5
MICROSLEEP_TIME = 3.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
UPPER_LIP, LOWER_LIP, LEFT_MOUTH, RIGHT_MOUTH = 13, 14, 78, 308
NOSE, CHIN, FOREHEAD = 1, 152, 10


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    return (a + b) / (2 * c)


def compute_mar(lm):
    vertical = distance((lm[UPPER_LIP].x, lm[UPPER_LIP].y), (lm[LOWER_LIP].x, lm[LOWER_LIP].y))
    horizontal = distance((lm[LEFT_MOUTH].x, lm[LEFT_MOUTH].y), (lm[RIGHT_MOUTH].x, lm[RIGHT_MOUTH].y))
    return vertical / max(horizontal, 1e-6)


def iris_center(lm, indices):
    pts = np.array([[lm[i].x, lm[i].y] for i in indices])
    return np.mean(pts, axis=0)


def gaze_ratio(lm, iris_idx, left_corner, right_corner):
    """Iris position inside the eye: about 0.5 = looking forward."""
    iris_x = np.mean([lm[i].x for i in iris_idx])
    a, b = lm[left_corner].x, lm[right_corner].x
    return (iris_x - a) / max(b - a, 1e-6)


class DriverMonitor:
    """Runs in webrtc worker thread. All state lives here, not in globals."""

    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.lock = threading.Lock()
        self.closed_start = None
        self.closed_frames = 0
        self.total_frames = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self.lock:
            results = self.detector.detect(mp_image)
            self.total_frames += 1

            if not results.face_landmarks:
                self.closed_start = None
                cv2.putText(img, "NO FACE", (20, 40), FONT, 1, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            lm = results.face_landmarks[0]

            # EAR
            left_eye = [(lm[i].x, lm[i].y) for i in LEFT_EYE]
            right_eye = [(lm[i].x, lm[i].y) for i in RIGHT_EYE]
            avg_ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2

            # Blendshapes
            left_blink = right_blink = 0
            if results.face_blendshapes:
                for item in results.face_blendshapes[0]:
                    if item.category_name == "eyeBlinkLeft":
                        left_blink = item.score
                    elif item.category_name == "eyeBlinkRight":
                        right_blink = item.score

            eye_closed = avg_ear < EAR_THRESHOLD or (
                left_blink > BLINK_THRESHOLD and right_blink > BLINK_THRESHOLD
            )

            duration = 0.0
            if eye_closed:
                self.closed_frames += 1
                if self.closed_start is None:
                    self.closed_start = time.time()
                duration = time.time() - self.closed_start
            else:
                self.closed_start = None

            if duration > MICROSLEEP_TIME:
                state = "MICROSLEEP"
            elif duration > DROWSY_TIME:
                state = "DROWSY"
            else:
                state = "NORMAL"

            # Yawn
            mar = compute_mar(lm)
            yawning = mar > MAR_THRESHOLD

            # Gaze
            g_left = gaze_ratio(lm, LEFT_IRIS, 33, 133)
            g_right = gaze_ratio(lm, RIGHT_IRIS, 362, 263)
            gaze_x = (g_left + g_right) / 2
            gaze = "FORWARD"
            if gaze_x < 0.40:
                gaze = "LOOKING LEFT"
            elif gaze_x > 0.60:
                gaze = "LOOKING RIGHT"

            # Head down / phone (relative to face size)
            face_h = max(lm[CHIN].y - lm[FOREHEAD].y, 1e-6)
            head_ratio = (lm[CHIN].y - lm[NOSE].y) / face_h
            head_down = head_ratio < 0.30
            phone = head_down and (0.40 < gaze_x < 0.60)

            perclos = self.closed_frames / max(self.total_frames, 1) * 100

        # Draw
        cv2.putText(img, f"EAR : {avg_ear:.2f}", (20, 40), FONT, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"MAR : {mar:.2f}", (20, 80), FONT, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"PERCLOS : {perclos:.1f}%", (20, 120), FONT, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Closed Time : {duration:.1f}s", (20, 160), FONT, 0.7, (255, 255, 0), 2)
        cv2.putText(img, gaze, (20, 200), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"HEAD RATIO : {head_ratio:.2f}", (20, 400), FONT, 0.7, (255, 255, 0), 2)

        if yawning:
            cv2.putText(img, "YAWNING DETECTED", (20, 240), FONT, 0.8, (0, 0, 255), 2)
        if head_down:
            cv2.putText(img, "HEAD DOWN", (20, 280), FONT, 0.8, (0, 0, 255), 2)
        if phone:
            cv2.putText(img, "PHONE DISTRACTION", (20, 320), FONT, 0.8, (0, 0, 255), 2)

        color = {"NORMAL": (0, 255, 0), "DROWSY": (0, 255, 255)}.get(state, (0, 0, 255))
        cv2.putText(img, state, (20, 370), FONT, 1, color, 3)

        # Visual alarm replaces winsound: red border when drowsy / microsleep
        if state != "NORMAL":
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# STREAMLIT UI

st.set_page_config(page_title="Driver Monitoring System", layout="centered")
st.title("Driver Monitoring System")
st.write("Click START and allow camera access in your browser.")

if not os.path.exists(MODEL_PATH):
    st.error("face_landmarker.task not found. Add it to the repo next to app.py.")
    st.stop()

webrtc_streamer(
    key="driver-monitor",
    video_processor_factory=DriverMonitor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
