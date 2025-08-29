import cv2
import os

def extract_frames_from_dataset(dataset_folder, save_folder, frame_skip=5):
    os.makedirs(save_folder, exist_ok=True)  

    video_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mp4")]

    if not video_files:
        print("No MP4 files found in the dataset folder!")
        return

    total_videos = len(video_files)
    processed_videos = 0

    for video_file in video_files:
        video_path = os.path.join(dataset_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  
        video_save_folder = os.path.join(save_folder, video_name)

        os.makedirs(video_save_folder, exist_ok=True) 

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        frame_count = 0
        saved_frames = 0
        success, image = cap.read()

        while success:
            if frame_count % frame_skip == 0:
                frame_filename = os.path.join(video_save_folder, f"frame_{frame_count}.jpg")
                if cv2.imwrite(frame_filename, image):
                    saved_frames += 1
                else:
                    print(f"Failed to save frame: {frame_filename}")

            frame_count += 1
            success, image = cap.read()

        cap.release()
        processed_videos += 1

        print(f"Processed {video_file}: {saved_frames} frames saved.")

    print(f"/nCompleted: Processed {processed_videos}/{total_videos} videos.")

dataset_path = "C:/Users/airob/OneDrive/Desktop/RSTM_Sampled/Videos"  
save_path = "C:/Users/airob/OneDrive/Desktop/Final_ExtractedFrames"  

extract_frames_from_dataset(dataset_path, save_path)
