import cv2
import mediapipe as mp
import csv
import copy
import itertools
import os
import string

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to calculate hand landmarks
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
    
    base_x, base_y = temp_landmark_list[0]  # Base reference point
    for index, (x, y) in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = x - base_x
        temp_landmark_list[index][1] = y - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list), default=1)  # Avoid division by zero
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list

# Function to log keypoints into CSV
def logging_csv(letter, landmark_list, csv_path="numAlpha.csv"):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([letter] + landmark_list)

# Path to dataset
dataset_path = ""

# Letters A-Z (folders in dataset)
alphabet = list(string.ascii_uppercase)
numbers = [str(i) for i in range(1, 10)]  # 1-9

# Process Images using MediaPipe Hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for letter in numbers:
        letter_folder = os.path.join(dataset_path, letter)  # Path to letter folder
        if not os.path.exists(letter_folder):
            print(f"⚠️ Skipping {letter_folder}: Folder not found")
            continue

        print(f"🔍 Processing images for letter '{letter}'...")

        # Dynamically get all image files in the folder
        image_files = [f for f in os.listdir(letter_folder) if f.endswith('.jpg')]

        for img_file in image_files:
            img_path = os.path.join(letter_folder, img_file)

            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Failed to load {img_path}, skipping...")
                continue

            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # If no hands detected, skip image
            if not results.multi_hand_landmarks:
                continue

            # Process detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)
                logging_csv(letter, processed_landmarks)

print("✅ Processing completed! Keypoints saved in keypoint.csv")
