import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#MODEL_PATH = "hand_landmarker.task" 
MODEL_PATH=r"C:\Users\mounilka\OneDrive\Desktop\demo\hand_landmarker.task"

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

#providing options to landmarker
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamp_ms = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    ) #converting frame to mp image

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )#applying landmark on mp image to detect hands

    h, w, _ = frame.shape

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            for landmark in hand_landmarks:
                x = int(landmark.x * w)#converting mp coordinates to pixel coordinates
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Hand Landmarks", frame)

    frame_timestamp_ms += 33

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()