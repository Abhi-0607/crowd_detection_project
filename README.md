
# Crowd Detection Project

This project leverages a YOLOv5 pre-trained object detection model to detect persons in a video and identify crowd events based on proximity and persistence. The program processes a video, detects persons in each frame, and logs crowd-related data into a CSV file.

---

## **Features**
- Detects persons in a video using YOLOv5.
- Identifies groups of three or more persons standing close to each other.
- Logs crowd events when groups persist for at least 10 consecutive frames.
- Outputs a CSV file containing frame numbers and the number of persons in the detected crowd.
- Displays video frames with detected bounding boxes.

---

## **Project Structure**
```
.
├── data/
│   └── sample_video.mp4     # Input video file
├── results/
│   └── crowd_detection_log.csv  # Output CSV file with crowd data
├── yolov5/                  # YOLOv5 model files (if cloned as a submodule)
├── main.py                  # Main script for the project
└── README.md                # Project documentation
```

---

## **Setup Instructions**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.8 or above
- pip (Python package manager)
- Git

### **2. Clone the Repository**
```bash
git clone https://github.com/Abhi-0607/crowd_detection_project.git
cd crowd_detection_project
```

### **3. Install Dependencies**
Set up a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## **How to Run the Project**

### **1. Place Your Video**
Add your input video file to the `data/` folder. Make sure the file path matches the one in `main.py`.

### **2. Run the Script**
Run the `main.py` file:
```bash
python main.py
```

### **3. View Results**
- The processed video frames will be displayed with bounding boxes around detected persons.
- Crowd events will be logged in the `results/crowd_detection_log.csv` file.

---

## **Crowd Detection Logic**
- **Definition**: A crowd is defined as three or more persons standing close to each other for 10 consecutive frames.
- **Steps**:
  1. Detect persons in each frame using YOLOv5.
  2. Calculate the distance between bounding boxes.
  3. Group persons based on proximity.
  4. Track groups across frames to detect persistent crowds.
  5. Log crowd events in a CSV file.

---

## **Dependencies**
- OpenCV
- NumPy
- PyTorch
- YOLOv5 (pre-trained model from Ultralytics)

---

## **Contributing**
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

---

## **License**
This project is licensed under the MIT License.

---

## **Acknowledgments**
- [Ultralytics](https://github.com/ultralytics/yolov5) for providing the YOLOv5 model.
- OpenAI for assisting with project development
