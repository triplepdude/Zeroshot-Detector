import os
import random
import cv2
from pathlib import Path

def parse_frame_data(txt_path):
    """Parse the frame classification data from the .txt file."""
    frame_data = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Frame") and "Label -" in line:
                try:
                    parts = line.split(":")
                    frame_number = int(parts[0].split()[1])
                    label = parts[1].split("Label -")[1].split(",")[0].strip().strip("'")
                    frame_data.append((frame_number, label))
                except (ValueError, IndexError):
                    continue
    return frame_data

def find_correct_path(base_dir, file_name):
    """Find the correct path for a given file name, handling subdirectories."""
    if file_name.startswith("_"):
        # Extract subdirectory name from the file name
        subdirectory = file_name.split("_")[1]
        return os.path.join(base_dir, subdirectory, file_name)
    return os.path.join(base_dir, file_name)

def extract_and_annotate_frame(video_path, frame_number, label, category, output_dir):
    """Extract and annotate a single frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        # Annotate the frame
        annotated_frame = cv2.putText(
            frame,
            label,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),  # Yellow text
            2,
            cv2.LINE_AA,
        )

        # Save the frame
        frame_filename = f"{category}_frame_{frame_number}_{Path(video_path).stem}.jpg"
        frame_filepath = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_filepath, annotated_frame)
    cap.release()

def main():
    categories = ["aircraft", "ground_vehicles", "maritime", "nonmilitary", "weapons"]
    base_dir = Path.cwd()
    output_dir = base_dir / "testing"
    output_dir.mkdir(exist_ok=True)

    required_frames_per_category = 100

    for category in categories:
        category_dir = base_dir / category
        all_files = []
        for root, _, files in os.walk(category_dir):
            for file in files:
                if file.endswith("_counts.txt"):
                    txt_path = find_correct_path(root, file)
                    video_path = txt_path.replace("_counts.txt", ".mp4")
                    if os.path.exists(video_path):
                        all_files.append((txt_path, video_path))

        total_extracted = 0
        selected_videos = set()

        while total_extracted < required_frames_per_category and len(selected_videos) < len(all_files):
            # Randomly pick a video
            txt_file, video_path = random.choice(all_files)
            if (txt_file, video_path) in selected_videos:
                continue
            selected_videos.add((txt_file, video_path))

            # Parse the frame data
            frame_data = parse_frame_data(txt_file)
            if not frame_data:
                continue

            # Select one frame from the video
            frame_number, label = random.choice(frame_data)
            extract_and_annotate_frame(video_path, frame_number, label, category, output_dir)
            total_extracted += 1

        print(f"{total_extracted} frames extracted for category {category}")


if __name__ == "__main__":
    main()
