import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#MODEL_PATH = "pose_landmarker.task"
MODEL_PATH=r"C:\Users\mounilka\OneDrive\Desktop\demo\pose_landmarker_full.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)#load the model

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)#parameters passing

detector = vision.PoseLandmarker.create_from_options(options) # creating landmark/detector

cap = cv2.VideoCapture(0)

timestamp = 0

# Pose skeleton connections
POSE_CONNECTIONS = [
    (11,13), (13,15),      # left arm
    (12,14), (14,16),      # right arm
    (11,12),               # shoulders
    (11,23), (12,24),      # torso
    (23,24),               # hips
    (23,25), (25,27),      # left leg
    (24,26), (26,28)       # right leg
]

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    h, w, _ = frame.shape

    if result.pose_landmarks:

        pose = result.pose_landmarks[0]

        points = []

        for landmark in pose:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw skeleton
        for start, end in POSE_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )

    cv2.imshow("Pose Detection", frame)

    timestamp += 33

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()