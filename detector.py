import cv2
import numpy as np
import easyocr
import re
import gradio as gr
from datetime import datetime, timedelta
from ultralytics import YOLO
from PIL import Image
import supervision as sv
from PIL import ImageDraw, ImageFont
import os

# Load the YOLO model
model_path = os.path.join(os.getcwd(), 'weights', 'best.pt')
model = YOLO(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Set area threshold for bounding box
AREA_THRESHOLD = 3000

# Regex for PAN number format
pan_regex = r"[A-Z]{5}[0-9]{4}[A-Z]"


# Helper function to process each frame
def process_frame(frame, stable_detections, last_detection_time, video_timestamp=None):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO model on the frame
    results = model.predict(rgb_frame, conf=0.4)
    result = results[0]  # Get the first (and only) result

    boxes = result.boxes

    # Variables to store results
    ocr_triggered = False
    ocr_text = ""
    pan_number = None  # To store the matched PAN number

    # Motion detection and stability check
    for box in boxes:
        x1y1x2y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = x1y1x2y2
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())

        # Get class name
        label = model.names[cls]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_area = (x2 - x1) * (y2 - y1)

        # Only proceed if the detected box area is above the threshold and label is "pan"
        if box_area > AREA_THRESHOLD and label == "pan":
            # Check stability by comparing with previous detections
            current_time = datetime.now()
            if label in stable_detections and (current_time - last_detection_time[label]).total_seconds() < 1:
                # If stable, perform OCR

                # Crop the detected region
                cropped = frame[y1:y2, x1:x2]

                # Check if cropped image is large enough
                if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    continue  

                # Calculate the variance of Laplacian (blurriness measure)
                gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray_cropped, cv2.CV_64F).var()

                # Set a threshold for blurriness
                BLUR_THRESHOLD = 50  

                if laplacian_var < BLUR_THRESHOLD:
                    # Image is too blurry, skip OCR
                    continue

                # Apply preprocessing steps to improve OCR results
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                # Apply adaptive thresholding, 'C' value is optimally chosen upon experimentation
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 7)

                # Run EasyOCR on the preprocessed cropped region
                ocr_result = reader.readtext(gray)
                ocr_text = " ".join([text[1] for text in ocr_result])

                print(f"ocr text: {ocr_text}")

                # Regex matching for PAN number
                pan_match = re.search(pan_regex, ocr_text)
                if pan_match:
                    pan_number = pan_match.group()
                    pan_number.upper()
                    ocr_triggered = True
                    break  # Only trigger OCR once per frame if detected

            # Update stable detection
            stable_detections[label] = (x1, y1, x2, y2)
            last_detection_time[label] = current_time

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)

        # Draw bounding box and label
        color = (0, 255, 0) if ocr_triggered else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 20)
        draw.text((x1, y1 - 10), f"{label} {pan_number if pan_number else ''}", fill=color, font=font)

        # Convert PIL Image back to OpenCV format
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


    return frame, ocr_triggered, video_timestamp if ocr_triggered else None, pan_number

# Function to process uploaded video
def process_uploaded_video(video):
    # Variables for stability checks
    stable_detections = {}
    last_detection_time = {}

    # Gradio outputs
    pan_numbers = []
    timestamps = []

    # Process the video frame by frame
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current timestamp in the video in milliseconds
        video_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        video_timestamp = str(timedelta(milliseconds=video_pos_ms))

        # Process each frame
        processed_frame, ocr_triggered, timestamp, pan_number = process_frame(
            frame, stable_detections, last_detection_time, video_timestamp)

        # If OCR was triggered, record the PAN number and timestamp
        if processed_frame is not None:
            if ocr_triggered and timestamp and pan_number:
                pan_numbers.append(pan_number)
                timestamps.append(timestamp)

    cap.release()

    # Return the last processed frame (or a default image), pan numbers, and timestamps
    if 'processed_frame' in locals():
        # Convert processed_frame to PIL Image for Gradio
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_frame_pil = Image.fromarray(processed_frame_rgb)
    else:
        processed_frame_pil = None  # Or a default image

    # Remove duplicates from pan_numbers and timestamps
    pan_numbers = list(set(pan_numbers))
    timestamps = list(set(timestamps))

    return processed_frame_pil, "\n".join(pan_numbers), "\n".join(timestamps)

# Gradio app with video upload
with gr.Blocks() as demo:
    gr.Markdown("# PAN Card Detection and OCR")
    with gr.Tabs():
        with gr.TabItem("Upload Video"):
            video_input = gr.Video(label="Upload Video")
            process_button = gr.Button("Process Video")
            video_image = gr.Image(label="Frame with Detection")
            pan_number_video = gr.Textbox(label="Detected PAN Numbers")
            timestamps_video = gr.Textbox(label="Timestamps")

            process_button.click(fn=process_uploaded_video, inputs=video_input, outputs=[video_image, pan_number_video, timestamps_video])

demo.launch()
