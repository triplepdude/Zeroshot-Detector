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
import multiprocessing

# MinIO configuration
client = Minio("whisper3:9000", secure=False)
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
def initialize_models():
    global model, processor, classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

# Set environment variable to disable parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load master roster to skip already processed files
processed_files = {}
if os.path.exists(master_roster_file):
    with open(master_roster_file, 'r') as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) == 2:
                processed_files[parts[0]] = parts[1]

# Avoid using tokenizers before the fork
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    initialize_models()

    # Loop through video files in MinIO
    try:
        objects = client.list_objects(bucket_name, recursive=True)
        category_totals = Counter()

        for obj in objects:
            # Check if the object is a video file and hasn't been processed
            if obj.object_name.endswith(('.mp4', '.avi', '.mov', '.mkv')) and obj.object_name not in processed_files:
                # Generate presigned URL for the video
                url = client.presigned_get_object(bucket_name, obj.object_name, expires=datetime.timedelta(hours=1))
                print(f'Processing video: {obj.object_name}, Access it at: {url}')
                
                # Download video
                video_filename = os.path.join(output_dirs['unknown'], obj.object_name)
                os.makedirs(os.path.dirname(video_filename), exist_ok=True)
                client.fget_object(bucket_name, obj.object_name, video_filename)

                # Open video and process frames
                cap = cv2.VideoCapture(video_filename)
                frame_interval = 30
                frame_count = 0
                label_counts = Counter()
                frame_classifications = []  # To store the classification for each frame
                non_military_count = 0  # Counter for non-military frames

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        inputs = processor(text="<OD>", images=pil_image, return_tensors="pt").to(device, torch_dtype)

                        try:
                            # Generate predictions with Florence-2
                            generated_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024,
                                num_beams=3,
                                do_sample=False
                            )
                            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                            # Classify the generated text into one of the categories
                            classification = classifier(generated_text, candidate_labels=labels)
                            category = "non_military"
                            if classification["scores"][0] >= 0.8:
                                category = classification["labels"][0]
                            else:
                                non_military_count += 1

                            # Ensure the category is in output_dirs to avoid KeyErrors
                            if category in output_dirs:
                                label_counts[category] += 1
                            else:
                                print(f"Warning: Detected unknown category '{category}' for '{obj.object_name}'.")
                                label_counts["unknown"] += 1

                            frame_classifications.append(f"Frame {frame_count}: Label - '{generated_text}', Classified as - '{category}'")
                        except torch.cuda.OutOfMemoryError:
                            print(f"Error processing frame {frame_count} of video '{obj.object_name}': CUDA out of memory. Skipping frame.")
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"Error processing frame {frame_count} of video '{obj.object_name}': {str(e)}. Skipping frame.")

                    frame_count += 1

                cap.release()

                # Ensure all categories have a count
                complete_label_counts = {label: label_counts.get(label, 0) for label in labels}
                most_frequent_category = max(complete_label_counts, key=complete_label_counts.get) if label_counts else 'unknown'

                if non_military_count > (frame_count / 2):
                    most_frequent_category = 'unknown'

                output_text_dir = os.path.join(output_dirs[most_frequent_category], os.path.dirname(obj.object_name))
                os.makedirs(output_text_dir, exist_ok=True)
                output_text_path = os.path.join(output_text_dir, f"{os.path.splitext(os.path.basename(obj.object_name))[0]}_counts.txt")
                
                with open(output_text_path, 'w') as f:
                    f.write("Frame classifications:\n")
                    for classification in frame_classifications:
                        f.write(classification + "\n")
                    f.write("\nSummary:\n")
                    for category, count in complete_label_counts.items():
                        f.write(f"{category}: {count}\n")

                output_video_dir = os.path.join(output_dirs[most_frequent_category], os.path.dirname(obj.object_name))
                os.makedirs(output_video_dir, exist_ok=True)
                output_video_path = os.path.join(output_video_dir, os.path.basename(obj.object_name))
                shutil.move(video_filename, output_video_path)
                print(f"Processed '{obj.object_name}' - categorized as '{most_frequent_category}' and moved to {output_dirs[most_frequent_category]}")

                with open(master_roster_file, 'a') as f:
                    f.write(f"{obj.object_name}, {most_frequent_category}\n")

                category_totals[most_frequent_category] += 1

    except S3Error as e:
        print(f'Error accessing MinIO: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')

    with open(master_roster_file, 'a') as f:
        f.write("\nSummary:\n")
        for category, total in category_totals.items():
            f.write(f"{category}: {total}\n")
    print("Master roster updated with summary of category totals.")
