
# Real-Time Demographic Analysis System

A Python-based computer vision application that performs real-time age, gender, and ethnicity estimation using webcam feed.

## Features

- **Real-time Face Detection** - Detects multiple faces simultaneously
- **Age Estimation** - Classifies age into 7 ranges with confidence scores
- **Gender Detection** - Identifies male/female with confidence scores
- **Ethnicity Recognition** - Classifies into 6 ethnic groups with balanced accuracy
- **Live Camera Feed** - Real-time processing with smooth performance

## Demo

![Demographic Analysis Demo](demo.gif)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/demographic-analysis.git
cd demographic-analysis
Create a virtual environment:

bash
python3 -m venv age_env
source age_env/bin/activate  # On Windows: age_env\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Download required model files:

bash
python3 download_model.py
python3 download_gender_model.py
Usage
Run the main application:

bash
python3 age_estimation.py
Controls:
Press 'q' to quit the application

Ensure good lighting for optimal accuracy

Face the camera directly for best results

Project Structure
text
demographic-analysis/
├── age_estimation.py          # Main application
├── download_model.py          # Downloads age/face detection models
├── download_gender_model.py   # Downloads gender model
├── create_missing_files.py    # Creates missing model config files
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
└── models/                    # Model files directory
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    └── opencv_face_detector.pbtxt
Models Used
Face Detection: OpenCV Haar Cascades

Age Estimation: Caffe model trained on Adience dataset

Gender Detection: Caffe model for binary classification

Ethnicity Estimation: Custom color-based analysis with multiple color spaces

Technical Details
Built with OpenCV and NumPy

Real-time processing at 20+ FPS

Multi-face detection and analysis

Confidence scoring for all predictions

Optimized for performance and accuracy

Limitations
Accuracy depends on lighting conditions and face angle

Ethnicity estimation is based on color analysis and may have limitations

Gender model requires proper training data for optimal accuracy

Performance may vary based on hardware

Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request
