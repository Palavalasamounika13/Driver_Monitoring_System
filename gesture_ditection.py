import cv2
import mediapipe as mp
from mediapipe.tasks import python #imports task api base classes
from mediapipe.tasks.python import vision

#MODEL_PATH = "hand_landmarker.task"
MODEL_PATH=r"C:\Users\mounilka\OneDrive\Desktop\demo\hand_landmarker.task"

# MediaPipe Hand Landmarker

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
) #load the model

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
) # pass the parameters to the model

detector = vision.HandLandmarker.create_from_options(options)

# Gesture Recognition

def recognize_gesture(hand_landmarks):

    tips = [4, 8, 12, 16, 20]

    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]

    index_tip = hand_landmarks[8]
    index_pip = hand_landmarks[6]

    middle_tip = hand_landmarks[12]
    middle_pip = hand_landmarks[10]

    ring_tip = hand_landmarks[16]
    ring_pip = hand_landmarks[14]

    pinky_tip = hand_landmarks[20]
    pinky_pip = hand_landmarks[18]

    fingers = []

    # Thumb
    if thumb_tip.x < thumb_ip.x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    fingers.append(1 if index_tip.y < index_pip.y else 0)
    fingers.append(1 if middle_tip.y < middle_pip.y else 0)
    fingers.append(1 if ring_tip.y < ring_pip.y else 0)
    fingers.append(1 if pinky_tip.y < pinky_pip.y else 0)

    # ------------------
    # Gesture Rules
    # ------------------

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"

    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"

    if fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"

    if fingers == [0, 1, 0, 0, 0]:
        return "POINTING"
    
    if fingers == [0, 1, 1, 1, 0]:
        return "THREE"

    return "UNKNOWN"


# ----------------------------
# Webcam
# ----------------------------

cap = cv2.VideoCapture(0)

timestamp = 0

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )#mediapipe image

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    h, w, _ = frame.shape

    if result.hand_landmarks: # to know the parameters passed are captured or not 

        for hand in result.hand_landmarks: # for every individual hand 

            # Draw landmarks
            for lm in hand:

                x = int(lm.x * w) # to convert into pixels
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (255, 255, 255),
                    -1
                )

            gesture = recognize_gesture(hand)

            cv2.putText(
                frame,
                gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("Gesture Recognition", frame)

    timestamp += 33

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()