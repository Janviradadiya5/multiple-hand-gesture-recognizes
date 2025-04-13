import cv2
import sys
import os
import math
import mediapipe as mp

# Suppress unnecessary stderr messages
sys.stderr = open(os.devnull, 'w')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Define gesture patterns
gesture_map = {
    (0, 0, 0, 0, 0): ("Fist", "✊"),
    (1, 1, 1, 1, 1): ("Open Palm", "🖐"),
    (0, 1, 1, 0, 0): ("Victory", "✌"),
    (1, 0, 0, 0, 0): ("Thumbs Up", "👍"),
    (1, 0, 0, 0, 1): ("Call Me", "🤙"),
    (1, 0, 1, 0, 1): ("Rock Sign", "🤘"),
    (0, 1, 0, 0, 0): ("Point", "🫵"),
    (0, 1, 1, 1, 0): ("Three Fingers", "🤟"),
    (0, 1, 1, 1, 1): ("Four Fingers", "🖖"),
   
    # "OK" is removed here — we'll handle it manually using distance
}

# Finger tips landmarks
tips = [4, 8, 12, 16, 20]

# Start webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()
else:
    print("Webcam opened successfully.")

# Function to get finger state
def get_finger_states(hand_landmarks):
    finger_states = []
    # Thumb
    finger_states.append(1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 2].x else 0)
    # Other fingers
    for i in range(1, 5):
        finger_states.append(1 if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y else 0)
    return tuple(finger_states)

# Function to calculate distance between two points
def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(handLms)

            # OK Sign Detection by Thumb-Index tip distance
            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]
            distance = get_distance(thumb_tip, index_tip)

            if distance < 0.04:
                gesture = ("OK Sign", "👌")
            else:
                gesture = gesture_map.get(fingers, ("Unknown", "❓"))

            text = f"{gesture[0]} {gesture[1]}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Display finger state
            cv2.putText(frame, str(fingers), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()