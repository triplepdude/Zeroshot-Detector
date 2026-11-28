#if running this for the first time in a session, you might have to pip install opencv-python and other stuff
import os
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

output_folder = 'test'
video_path = 'test_tiktok.mp4'
cap = cv2.VideoCapture(video_path)
frame_interval = 30  # Capture every 30th frame (roughly like 1 per second at tiktok quality?)
frames = []
labels = []
frame_count = 0
device = "cuda"
torch_dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True) #basic pretrained model 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        frames.append(frame)  # Store the frame
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text="<OD>", images=pil_image, return_tensors="pt").to(device, torch_dtype)
    #doing object detection 
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        ) #the above code generates predictions based on pixel input (using basic options as specified in documentation for model
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        labels.append(generated_text) 
    frame_count += 1
cap.release()
#this part of the code onword I will remove because there is no need to save image with label on it for production variant.
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" 
font_size = 24 
font = ImageFont.truetype(font_path, font_size)

for i, (frame, label) in enumerate(zip(frames, labels)):
    print(label)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    text_position = (10, 10)  
    draw.text(text_position, label, font=font, fill=(0, 0, 0))  
    labeled_frame_path = os.path.join(output_folder, f'frame_{i}.jpg')
    pil_image.save(labeled_frame_path)
    print(f"Saved labeled frame: {labeled_frame_path}")
