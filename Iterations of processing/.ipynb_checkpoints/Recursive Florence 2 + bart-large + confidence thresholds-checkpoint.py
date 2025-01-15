import os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from collections import Counter
from minio import Minio
from minio.error import S3Error
import datetime
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader

# MinIO configuration
client = Minio(
    "whisper3:9000",
    secure=False
)
bucket_name = 'dfp-viral-moment-videos'

# Directories
output_dirs = {
    'weapons': 'weapons',
    'aircraft': 'aircraft',
    'maritime': 'maritime',
    'ground_vehicles': 'ground_vehicles',
    'non_military': 'non_military',
    'unknown': 'unknown'
}
for dir_name in output_dirs.values():
    os.makedirs(dir_name, exist_ok=True)

# Master roster file
master_roster_file = 'master_roster.txt'

# Initialize Florence-2 and classification models
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
labels = ["weapons", "aircraft", "maritime", "ground_vehicles", "non_military"]

# Set a confidence threshold for military-related classification
CONFIDENCE_THRESHOLD = 0.8

# Load master roster to skip already processed files
processed_files = {}
if os.path.exists(master_roster_file):
    with open(master_roster_file, 'r') as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) == 2:
                processed_files[parts[0]] = parts[1]

# Frame Processing Dataset
class VideoFrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        pil_image = Image.fromarray(cv2.cvtColor(self.frames[idx], cv2.COLOR_BGR2RGB))
        return pil_image

def classify_video(obj):
    # Check if the object is a video file and hasn't been processed
    if not (obj.object_name.endswith(('.mp4', '.avi', '.mov', '.mkv')) and obj.object_name not in processed_files):
        return None

    # Generate presigned URL for the video
    url = client.presigned_get_object(bucket_name, obj.object_name, expires=datetime.timedelta(hours=1))
    print(f'Processing video: {obj.object_name}, Access it at: {url}')
    
    # Download video
    video_filename = os.path.join(output_dirs['unknown'], obj.object_name)
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)  # Ensure nested directories are created
    client.fget_object(bucket_name, obj.object_name, video_filename)

    try:
        # Open video and process frames
        cap = cv2.VideoCapture(video_filename)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_filename}. Skipping.")
            return None

        frame_interval = 30
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append(frame)  # Collect frames for batch processing
            frame_count += 1

        cap.release()

        if not frames:
            print(f"Warning: No frames extracted from {video_filename}. Skipping.")
            return None

        # Use DataLoader for efficient batch processing on GPU
        dataset = VideoFrameDataset(frames)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True)

        label_counts = Counter()
        frame_classifications = []  # To store the classification for each frame

        for batch in dataloader:
            inputs = processor(text=["<OD>"] * len(batch), images=batch, return_tensors="pt").to(device, torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Classify each generated description
            for i, generated_text in enumerate(generated_texts):
                classification = classifier(generated_text, candidate_labels=labels)
                top_label = classification["labels"][0]
                top_score = classification["scores"][0]

                # Determine final category based on confidence threshold
                if top_score >= CONFIDENCE_THRESHOLD and top_label != "non_military":
                    final_category = top_label
                else:
                    final_category = "non_military"

                # Count the classification
                label_counts[final_category] += 1

                # Store frame-level classification
                frame_classifications.append(f"Frame {frame_count}: Label - '{generated_text}', Classified as - '{final_category}', Confidence - {top_score:.2f}")

        # Summarize and save results
        complete_label_counts = {label: label_counts.get(label, 0) for label in labels}
        most_frequent_category = max(complete_label_counts, key=complete_label_counts.get) if label_counts else 'unknown'

        # Write frame classifications and counts to a text file
        output_text_dir = os.path.join(output_dirs[most_frequent_category], os.path.dirname(obj.object_name))
        os.makedirs(output_text_dir, exist_ok=True)  # Ensure all directories exist
        output_text_path = os.path.join(output_text_dir, f"{os.path.splitext(os.path.basename(obj.object_name))[0]}_counts.txt")
        with open(output_text_path, 'w') as f:
            f.write("Frame classifications:\n")
            for classification in frame_classifications:
                f.write(classification + "\n")
            f.write("\nSummary:\n")
            for category, count in complete_label_counts.items():
                f.write(f"{category}: {count}\n")

        # Move the video to the corresponding folder and delete the local copy
        output_video_dir = os.path.join(output_dirs[most_frequent_category], os.path.dirname(obj.object_name))
        os.makedirs(output_video_dir, exist_ok=True)
        output_video_path = os.path.join(output_video_dir, os.path.basename(obj.object_name))
        shutil.move(video_filename, output_video_path)
        print(f"Processed '{obj.object_name}' - categorized as '{most_frequent_category}' and moved to {output_dirs[most_frequent_category]}")
        os.remove(video_filename)  # Delete the downloaded file
        return most_frequent_category

    except Exception as e:
        print(f"Error processing video {video_filename}: {e}")
        return None

# Main processing with concurrency
try:
    objects = client.list_objects(bucket_name, recursive=True)
    category_totals = Counter()

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(classify_video, obj) for obj in objects]
        for future in as_completed(futures):
            result = future.result()
            if result:
                category_totals[result] += 1

except S3Error as e:
    print(f'Error accessing MinIO: {e}')
except Exception as e:
    print(f'An error occurred: {e}')

# Final update to master roster with totals
with open(master_roster_file, 'a') as f:
    f.write("\nSummary:\n")
    for category, total in category_totals.items():
        f.write(f"{category}: {total}\n")
print("Master roster updated with summary of category totals.")
