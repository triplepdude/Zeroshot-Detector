import os
import cv2
import torch
import logging
import argparse
import datetime
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm
from collections import Counter
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from minio import Minio
from minio.error import S3Error
from io import BytesIO

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # warnings disabled
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(  #so i know what happens when stuff doesnt work
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
MINIO_ENDPOINT = #INPUT STRING HERE.
MINIO_SECURE = False #making it wasy to change buckets, currently 1 bucket per script
BUCKET_NAMES = [
    #LIST BUCKET NAMES HERE.
]
OUTPUT_DIRS = {
    'weapons': 'weapons',
    'aircraft': 'aircraft',
    'maritime': 'maritime',
    'ground_vehicles': 'ground_vehicles',
    'nonmilitary': 'nonmilitary'
}
LABELS = ["weapons", "aircraft", "maritime", "ground_vehicles", "nonmilitary"]
MASTER_ROSTER_FILE = 'master_roster.txt'
FRAME_INTERVAL = 30  # ~1 every second

def initialize_worker():
    global processor, model, classifier, device, torch_dtype, client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #attempt to use gpu before cpu
    torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained( #original params for Florence2 model
        "microsoft/Florence-2-large",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained( #double checking processor outputs are correctly achieved.
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli", #from docs
        device=0 if torch.cuda.is_available() else -1
    )
    client = Minio(MINIO_ENDPOINT, secure=MINIO_SECURE) #actually connect to bucket

def load_processed_files(master_roster_file):
    processed_files = {}
    if os.path.exists(master_roster_file): #reads master file and checks files previously processed
        with open(master_roster_file, 'r') as f:
            for line in f:
                parts = line.strip().split(", ")
                if len(parts) == 2:
                    processed_files[parts[0]] = parts[1]
    return processed_files

def process_video(obj, bucket_name):
    global processor, model, classifier, device, torch_dtype, client
    labels = LABELS
    output_dirs = OUTPUT_DIRS
    frame_interval = FRAME_INTERVAL
    master_roster_file = MASTER_ROSTER_FILE

    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)

    processed_files = load_processed_files(master_roster_file)

    if obj.object_name.endswith(('.mp4', '.avi', '.mov', '.mkv')) and obj.object_name not in processed_files: #only process video file extensions (but have only tried with mp4s...
        try:
            response = client.get_object(bucket_name, obj.object_name)
            video_data = BytesIO(response.read())
            response.close()
            response.release_conn()

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_data.getbuffer())
                temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            frame_count = 0
            label_counts = Counter({label: 0 for label in labels})  # Initialize counts for each label
            frame_classifications = []
            nonmilitary_count = 0
            prev_frame = None
            threshold = 5000
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Military keywords indicating ground vehicles
            military_keywords = [
                "armored", "convoy", "military", "humvee", "APC", "tank", "infantry",
                "MRAP", "Stryker", "Bradley", "military truck", "artillery",
                "howitzer", "patrol vehicle", "combat vehicle", "troop carrier",
                "military convoy", "tracked vehicle"
            ]

            with tqdm(total=total_frames, desc=f"Processing {obj.object_name}") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        if prev_frame is None or cv2.norm(frame, prev_frame, cv2.NORM_L1) > threshold: #weird fix for frame
                            prev_frame = frame
                            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                            # Object detection
                            inputs = processor(text="<OD>", images=pil_image, return_tensors="pt").to(device)

                            try:
                                # actual predictions
                                generated_ids = model.generate(
                                    input_ids=inputs["input_ids"].to(device),
                                    pixel_values=inputs["pixel_values"].to(device),
                                    max_new_tokens=1024,
                                    num_beams=3,
                                    do_sample=False
                                )
                                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                                if generated_text and isinstance(generated_text, str):
                                    # Check for military keywords in the generated text
                                    text_lower = generated_text.lower()
                                    if any(word in text_lower for word in military_keywords):
                                        category = "ground_vehicles"  # Ensure consistency with the key in `label_counts`
                                    else:
                                        # Use the classifier if no military keywords are detected
                                        classification = classifier(generated_text, candidate_labels=labels)
                                        if classification["scores"][0] >= 0.8:
                                            category = classification["labels"][0]
                                        else:
                                            category = "nonmilitary"
                                            nonmilitary_count += 1

                                    # Update `label_counts` for the identified category
                                    label_counts[category] += 1

                                    frame_classifications.append(f"Frame {frame_count}: Label - '{generated_text}', Classified as - '{category}'")
                                else:
                                    logging.warning(f"Invalid generated text for frame {frame_count} in video '{obj.object_name}'.")
                                    nonmilitary_count += 1

                            except torch.cuda.OutOfMemoryError:
                                logging.error(f"CUDA out of memory while processing frame {frame_count} of video '{obj.object_name}'. Skipping frame.")
                                torch.cuda.empty_cache()
                            except Exception as e:
                                logging.error(f"Error processing frame {frame_count} of video '{obj.object_name}': {str(e)}", exc_info=True)
                                nonmilitary_count += 1

                    frame_count += 1
                    pbar.update(1)

            cap.release()
            os.remove(temp_video_path)

            complete_label_counts = {label: label_counts[label] for label in labels}
            most_frequent_category = max(complete_label_counts, key=complete_label_counts.get) if label_counts else 'nonmilitary'#sets correct category

            if nonmilitary_count > (frame_count * 0.5): #non military more than half then make vid classed nonmilitary
                most_frequent_category = 'nonmilitary'

            output_text_dir = os.path.join(output_dirs[most_frequent_category], os.path.dirname(obj.object_name))
            os.makedirs(output_text_dir, exist_ok=True)
            output_text_path = os.path.join(output_text_dir, f"{os.path.splitext(os.path.basename(obj.object_name))[0]}_counts.txt") #updates output dir and master file.

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

            with open(output_video_path, 'wb') as f_out:
                f_out.write(video_data.getbuffer())

            logging.info(f"Processed '{obj.object_name}' - categorized as '{most_frequent_category}' and saved to '{output_video_path}'")

            with open(master_roster_file, 'a') as f:
                f.write(f"{obj.object_name}, {most_frequent_category}\n")

        except Exception as e:
            logging.error(f"Error processing video '{obj.object_name}': {str(e)}", exc_info=True) #from chatgpt to be more verbose with exceptions
    else:
        logging.info(f"Video '{obj.object_name}' has already been processed. Skipping.")

def process_bucket(bucket_name): #goes through buckets, somewhat obselete now
    global client
    client = Minio(MINIO_ENDPOINT, secure=MINIO_SECURE)

    if not client.bucket_exists(bucket_name):
        logging.warning(f"Bucket '{bucket_name}' does not exist.")
        return

    logging.info(f"Processing bucket: {bucket_name}")
    try:
        try:
            objects = list(client.list_objects(bucket_name, recursive=True))
        except Exception as e:
            logging.error(f"Error listing objects in bucket '{bucket_name}': {str(e)}", exc_info=True)
            return

        video_objects = [obj for obj in objects if obj.object_name.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        for obj in video_objects:
            process_video(obj, bucket_name)

    except S3Error as e:
        logging.error(f'Error accessing MinIO for bucket {bucket_name}: {e}', exc_info=True)
    except Exception as e:
        logging.error(f'An error occurred with bucket {bucket_name}: {e}', exc_info=True)

def main():
    parser = argparse.ArgumentParser(description='Process videos from MinIO buckets.')
    parser.add_argument('--buckets', nargs='+', help='List of bucket names to process', default=BUCKET_NAMES)
    args = parser.parse_args()
    initialize_worker()
    for bucket_name in args.buckets:
        if bucket_name:
            process_bucket(bucket_name)
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
