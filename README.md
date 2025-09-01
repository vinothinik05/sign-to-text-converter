Sign to Text Converter

Project Overview

This project converts sign language gestures into text using computer vision and machine learning techniques. It helps people with hearing or speech disabilities communicate more easily with others by detecting hand gestures from a camera feed and translating them into readable text.

Project Structure

Sign-to-Text-Converter/
│
├── main.py → Entry point – runs the sign to text converter
├── sign_capture.py → Captures sign gestures via webcam
├── preprocess.py → Preprocesses gesture images for recognition
├── model.py → Machine learning model for gesture recognition
├── predict.py → Maps recognized signs to text output
│
├── requirements.txt → Python dependencies
├── demo_video.mp4 → Demo video of the project (or YouTube link below)
├── README.md → Documentation file
│
├── assets/ → Supporting files
│ ├── dataset/ → Training dataset of sign images
│ ├── trained_model/ → Saved ML model weights
│ └── outputs/ → Predicted text results
│
└── utils/ → Helper functions and scripts

Installation

Step 1: Clone the Repository
git clone https://github.com/your-username/sign-to-text-converter.git

cd sign-to-text-converter

Step 2: Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)

Step 3: Install Dependencies
pip install -r requirements.txt

Usage

Run the main program:
python main.py

Workflow:

1. Show sign gestures in front of the webcam.
2. The system detects the hand/gesture.
3. The ML model recognizes the sign.
4. Corresponding text is displayed on the screen.

Demo


https://github.com/user-attachments/assets/a5c534c9-4b5c-43b5-95ea-2f601209050b


Contribution

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Submit a pull request
