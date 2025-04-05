import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("alnum_tuvwxyzski.h5")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


alphabet_subset =  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def calc_landmark_list(image, landmarks):
    """Convert landmarks to image coordinates"""
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(lm.x * image_width), image_width - 1),
             min(int(lm.y * image_height), image_height - 1)] 
            for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    """Normalize landmarks exactly like during training"""
    # Convert to numpy array
    landmarks = np.array(landmark_list, dtype=np.float32)
    
    # Center around wrist (landmark 0)
    landmarks -= landmarks[0]
    
    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    
    return landmarks.reshape(1, 21, 2, 1)  # Match model input shape

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,  # Reduced complexity for faster processing
    max_num_hands=2,     # Only detect one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        # Mirror display
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get and preprocess landmarks
                landmarks = calc_landmark_list(image, hand_landmarks)
                processed = pre_process_landmark(landmarks)
                
                # Get prediction
                predictions = model.predict(processed, verbose=0)[0]
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)
                
                # Only show if confidence > 70%
                if confidence > 0.7:
                    label = f"{alphabet_subset[predicted_class]} ({confidence:.2f})"
                    cv2.putText(image, label, (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('Indian Sign Language Detector', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()