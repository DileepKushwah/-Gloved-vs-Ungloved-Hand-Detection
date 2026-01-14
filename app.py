import streamlit as st
import numpy as np
import os
import json
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "best_glove_model.pt"
CONFIDENCE_THRESHOLD = 0.5

OUTPUT_DIR = "streamlit_outputs"
LOG_DIR = "streamlit_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🧤 Glove Detection System")
st.write("Detect **gloved hand** and **bare hand** from images or videos.")

option = st.radio("Select input type:", ["Image", "Video"])

# -------------------------
# IMAGE DETECTION
# -------------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        import cv2  # ✅ LAZY IMPORT (IMPORTANT)

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)

        detections = []
        annotated = img.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2]
                })

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {confidence:.2f}",
                    (x1, max(y1 - 7, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        st.image(annotated, channels="BGR", caption="Detection Result")

        json_path = os.path.join(LOG_DIR, uploaded_file.name + ".json")
        with open(json_path, "w") as f:
            json.dump({
                "filename": uploaded_file.name,
                "detections": detections
            }, f, indent=2)

        st.success("Detection completed and JSON saved")

# -------------------------
# VIDEO DETECTION
# -------------------------
if option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        import cv2  # ✅ LAZY IMPORT (IMPORTANT)

        temp_video_path = os.path.join(OUTPUT_DIR, uploaded_video.name)

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(y1 - 7, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("Video processing completed")
