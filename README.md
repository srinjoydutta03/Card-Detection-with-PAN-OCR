# PAN Card Detection and OCR

A Gradio-based application that uses YOLO for detecting PAN cards and Aadhar Cards and EasyOCR for extracting PAN numbers from videos. The application leverages a stability check and bounding box area threshold to ensure OCR is only performed on stable frames with significant content, improving both accuracy and efficiency. The YOLO11s model was fine-tuned using a dataset available at Roboflow.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PAN Card Detection**: Uses a YOLO11s model to detect PAN cards in video frames.
- **OCR with EasyOCR**: Extracts PAN numbers from detected regions.
- **Detected Bounding Box Size Check**: Checks whether the bounding box is above threshold, only then considered for OCR.
- **Stability Check**: OCR is triggered only on stable detections, reducing redundant OCR calls and improving accuracy.
- **Preprocessing before OCR**: Adaptive thresholding is done by converting the cropped region after converting it into grayscale, carefully selected hyperparameters by experimentation for optimal OCR Performance.
- **Timestamp Tracking**: Records the exact timestamp in the video when each PAN number is detected.
- **Video and Webcam Support**: The app supports both video uploads and recording input.

## Installation

### Prerequisites
- Python 3.10.x (this is due to some packages not supporting newer python versions)
- `pip` for Python package management

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/srinjoydutta03/Card-Detection-with-PAN-OCR.git
   cd Card-Detection-with-PAN-OCR/
   ```

2. **Set Up a Virtual Environment**
   It’s recommended to use a virtual environment to avoid dependency conflicts.
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Packages**
   Install the packages specified in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO Model Weights**
   - The YOLO model weights (`best.pt`) is in the `weights` directory and is ready to use. If you have trained the model yourself, place the file in `weights/best.pt`.

5. **Font for Displaying Text on Frames**
   - On macOS, Arial is typically located at `/Library/Fonts/Arial.ttf`. For other OSes, update the font path in the `process_frame` function if necessary.

## Usage

1. **Start the Application**
   Run the following command to launch the Gradio app:
   ```bash
   python3 detector.py
   ```

2. **Using the App**
   - **Upload Video**: Use the "Upload Video" tab to upload a video file or record for processing (Note: Recording may inverse the video resulting in detection and classification but extraction is impacted).

3. **Outputs**
   - **Detected PAN Number**: Displays the PAN numbers detected in the video.
   - **Timestamps**: Shows the timestamps of detected PAN numbers in the video.
   - **Logs**: The terminal displays the logs of the type of card detected and whether OCR was performed or not

## Screenshots

Screenshots of the app interface and outputs will be added here.

- **App Interface**  
  ![App Interface](assets/app_interface.png)

- **Processed Frame with Detection**  
  ![Processed Frame](assets/processed_frame.png)

- **Detected PAN Number and Timestamp**  
  ![PAN Number and Timestamp](assets/pan_number_timestamp.png)

## Project Structure

```
PAN-CARD-WITH-OCR/
├── .venv/                  # Virtual environment
├── train-yolo/             # Training scripts and configuration for YOLO model (if any)
├── weights/
│   └── best.pt             # YOLO model weights for PAN card detection
├── detector.py             # Main application script
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
└── assets/                 # Folder for storing screenshots and other assets
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.


## License

This project is licensed under the MIT License.
```