# -Gloved-vs-Ungloved-Hand-Detection
 Gloved vs Ungloved Hand Detection using yolov8 of image dataset 
Project Overview
This project is developed to detect whether a hand is wearing a glove or not using computer vision.
The system can identify and classify hands into two categories:
1.	bare_hand
2.	gloved_hand
The model works on both images and videos and also provides the location of the hand using rectangular bounding boxes.

Dataset
Dataset Source: 
The dataset was collected from Roboflow, using publicly available YOLOv8-compatible image detection datasets related to hands and gloves.
Dataset Format:
YOLO format (images + .txt label files)
Split into:
1.	train
2.	valid
Classes Used:
1.	bare_hand
2.	gloved_hand
The dataset was downloaded directly using the Roboflow API and used without manual relabeling.

Model Used
•	YOLOv8 (Ultralytics)
•	Base model: yolov8n.pt
•	Framework: PyTorch
•	Training and testing done using Ultralytics YOLOv8
The model was fine-tuned to detect only two classes: gloved hand and bare hand.

Training Environment
•	Platform: Google Colab
•	Python Version: 3.x
•	Libraries Used:
o	ultralytics
o	opencv-python
o	matplotlib
o	numpy
Google Colab was used for faster training and easy GPU access.

Dataset Preprocessing
Before training, basic preprocessing was performed to clean the dataset.
Removed Empty Label Files
Some images had empty .txt label files.
This helped avoid training errors and improved model stability.

Training Process
•	Model was trained using YOLOv8 training pipeline
•	Image size: 640 x 640
•	Epochs: 30
•	Batch size: 16
•	Default YOLOv8 augmentations were used
The best trained model was saved as:
best_glove_model.pt


Inference (Testing)
The trained model can detect:
•	bare_hand
•	gloved_hand
Features:
•	Works on images
•	Works on videos
•	Draws rectangular bounding boxes
•	Saves:
o	Annotated images/videos – output folder
o	JSON logs with detection details – log folder
Each detection includes:
•	Class label
•	Confidence score
•	Bounding box coordinates

How to Run the Project
Run on Images (Google Colab)
1.	Upload:
o	best_glove_model.pt
o	detect.py
o	Folder with test images (test folder)
2.	Run:
Output:
•	output→ annotated images
•	logs → JSON files for each image
What Worked Well
•	YOLOv8 detected gloves and bare hands accurately
•	Bounding boxes were properly aligned
•	Model worked well on both images and videos
•	Training was stable after cleaning empty labels

What Did Not Work Well
•	Very small hands were sometimes missed
•	Gloves with skin-like color reduced confidence
•	Low-light images affected accuracy slightly
