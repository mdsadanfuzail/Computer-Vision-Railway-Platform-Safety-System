# Computer Vision Railway Platform Safety System

The Railway Platform Safety System is a computer vision-based solution designed to ensure the safety of people on railway platforms. The system detects the presence of people, segments railway tracks, and checks the proximity of people to tracks to identify potential danger zones. It uses YOLOv8 for object detection and Fine-tuned YOLOv11n-seg model for segmentation, with OpenCV for safety checks and visualization.

## Features

- **Person Detection**: Detects people on the railway platform using YOLOv5s.
- **Track Segmentation**: Segments and identifies tracks using a custom Fine-tuned YOLOv11n-seg model.
- **Proximity Check**: Monitors the proximity of people to the tracks and raises a danger warning if the distance is too close.
- **Visualization**: Displays bounding boxes around detected people and track segmentation overlays on video frames.
- **Real-Time Processing**: Processes video frames in real-time to ensure timely detection of safety risks.

## Fine-Tuining YOLOv11n-seg Model on Google Colab

You can view the Fine-Tuning YOLOv11n-seg model notebook on Google Colab [here](https://colab.research.google.com/drive/16YhaC_rWVklu9hW_AzIg5Wm2sM3BjLa1?usp=sharing).

## Tech Stack

- **Programming Language**: Python
- **Libraries/Frameworks**:
  - [YOLOv5](https://github.com/ultralytics/yolov5) for object detection
  - [YOLOv11](https://github.com/ultralytics/ultralytics) for segmentation
  - [PyTorch](https://pytorch.org/) for model loading, inference, and training
  - [OpenCV](https://opencv.org/) for image processing and visualization
- **Model Weights**: Fine-tuned YOLOv11n-seg and pre-trained YOLOv5s model weights
- **Video Processing**: Uses OpenCV's `cv2.VideoCapture` for video input
- **Detection and Segmentation**: YOLOv5s for detecting people and Fine-tuned YOLOv11n-seg for segmenting tracks

## Citation

@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}


## Installation

To run the Railway Platform Safety System locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Railway-Platform-Safety-System.git
   ```

2. Navigate to the project folder:
   ```bash
   cd Railway-Platform-Safety-System
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Ensure all required models and dependencies are installed.
2. Run the main application:
   ```bash
   python main.py
   ```

3. The system will start processing the video feed and display the results in real-time.

4. To exit, press `q`.

## Project Structure

```
Computer-Vision-Railway-Platform-Safety-System/
|
├── assets/                          # Video files and other assets
├── weights/                         # YOLOv8 and YOLOv11 model weights (pre-trained and fine-tuned)
├── Fine-Tuning YOLOv11n-seg.ipynb   # Notebook for fine-tuning YOLOv11n-seg
├── code/                            
│   ├── main.py                      # Entry point of the system, handles video processing and safety checks
│   ├── person_detection.py          # Person detection using YOLOv8n
│   ├── track_segmentation.py        # Segments railway tracks using YOLOv11n-seg
│   ├── proximity_check.py           # Checks distance of people from tracks and identifies danger zones
│   ├── visualization.py             # Visualizes the results and overlays on video frames
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to YOLOv8 for providing the powerful object detection and segmentation models used in this project.
- Thanks to OpenCV for enabling efficient image processing and visualization capabilities.