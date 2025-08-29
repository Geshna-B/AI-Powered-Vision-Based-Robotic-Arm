import mediapipe as mp
import numpy as np
import cv2
import os
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_keypoints(image_path):
    """Extracts 21 hand keypoints (x, y) from an image using Mediapipe."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append(landmark.x)
                keypoints.append(landmark.y)

    return np.array(keypoints) if keypoints else np.zeros(42)  

def process_dataset(image_folder, save_json="hand_keypoints.json"):
    """Extracts keypoints from all images in the dataset and saves as JSON."""
    keypoints_dict = {}

    for video_folder in os.listdir(image_folder):  
        video_path = os.path.join(image_folder, video_folder)
        
        if os.path.isdir(video_path):  
            keypoints_dict[video_folder] = []

            for image_file in os.listdir(video_path):  
                if image_file.endswith(".jpg"): 
                    image_path = os.path.join(video_path, image_file)
                    keypoints = extract_keypoints(image_path)
                    keypoints_dict[video_folder].append(keypoints.tolist())  

    with open(save_json, "w") as f:
        json.dump(keypoints_dict, f)

dataset_path = "C:/Users/airob/OneDrive/Desktop/Final_ExtractedFrames"  
process_dataset(dataset_path)
