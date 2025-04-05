import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string

# Load the trained CNN model
model = keras.models.load_model("Islcnn4epoch.h5")

# Initialize Mediapipe Hands model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define alphabet labels
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# Function to extract landmark coordinates
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to normalize landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(map(abs, temp_landmark_list)) if max(map(abs, temp_landmark_list)) != 0 else 1
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return np.array(temp_landmark_list).reshape(1, 21, 2, 1)  # Reshape for CNN model

# Open webcam for real-time detection
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)  # Flip for a selfie view
        
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)

                # Predict sign language
                predictions = model.predict(processed_landmarks, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                
                label = alphabet[predicted_class]
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                print(label)
                print("------------------------")

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Show output
        cv2.imshow('Indian Sign Language Detector', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
