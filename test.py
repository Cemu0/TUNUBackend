# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='model/sign_language_recognizer_25-04-2023.task')
# base_options = python.BaseOptions(model_asset_path='model/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


def display_gestures_and_hand_landmarks(image, result):
    """Displays the image with the gesture category and its score along with the hand landmarks."""
    # Image and labels.
    gesture = result[0]
    hand_landmarks = result[1]

    # Display gesture and hand landmarks.
    annotated_image = image.copy()
    title = f"{gesture.category_name} ({gesture.score:.2f})"
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

    return annotated_image

# STEP 3: load camera feed
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the image horizontally for a later selfie-view display.
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process the frame and get hand landmarks.
    results = recognizer.recognize(mp_image)

    if results.gestures:

        top_gesture = results.gestures[0][0]
        hand_landmarks = results.hand_landmarks[0]
                
        # Display the image with the gesture category and its score along with the hand landmarks.
        annotated_image = display_gestures_and_hand_landmarks(frame, (top_gesture, hand_landmarks))
        
        print (top_gesture.category_name, top_gesture.score)

        cv2.imshow('MediaPipe Gesture Recognition', annotated_image)
    else:
        cv2.imshow('MediaPipe Gesture Recognition', frame)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break