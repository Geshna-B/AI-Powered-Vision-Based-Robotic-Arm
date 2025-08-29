import os
import random
import shutil
import pickle
import gzip

video_dir = "C:/Users/airob/OneDrive/Desktop/archive (1)/videos_phoenix/videos/train"  
annotations_file = "C:/Users/airob/OneDrive/Desktop/archive (1)/phoenix14t.pami0.train.annotations_only.gzip"  
output_video_dir ="C:/Users/airob/OneDrive/Desktop/RSTM_Sampled/Videos" 
output_annotations_file = "C:/Users/airob/OneDrive/Desktop/RSTM_Sampled/sampled_annotations.gzip"  

os.makedirs(output_video_dir, exist_ok=True)

video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

random.seed(42)  
sampled_videos = random.sample(video_files, int(0.3 * len(video_files)))

for video in sampled_videos:
    shutil.copy(os.path.join(video_dir, video), os.path.join(output_video_dir, video))

with gzip.open(annotations_file, 'rb') as f:
    annotations = pickle.load(f)

sampled_annotations = [ann for ann in annotations if ann['name'] in sampled_videos]

with gzip.open(output_annotations_file, 'wb') as f:
    pickle.dump(sampled_annotations, f)

print(f"Sampled {len(sampled_videos)} videos and saved their annotations.")
