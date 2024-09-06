import streamlit as st
import cv2
import face_recognition
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import defaultdict
from PIL import Image
import pickle
import os

# Load the pre-trained Huggingface model for age classification
@st.cache_resource
def load_age_model():
    processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
    return processor, model

processor, model = load_age_model()

# Dictionary to store known faces and their encodings
known_faces = defaultdict(list)
face_data = defaultdict(dict)
face_id = 0

# File to store persistent data
PERSISTENT_FILE = 'face_recognition_data.pkl'

# Load persistent data if it exists
if os.path.exists(PERSISTENT_FILE):
    with open(PERSISTENT_FILE, 'rb') as f:
        data = pickle.load(f)
        known_faces = data['known_faces']
        face_data = data['face_data']
        face_id = data['face_id']

# Convert OpenCV image to PIL format
def preprocess_image(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_img, return_tensors="pt")
    return inputs

# Predict age using Hugging Face model
def predict_age(face_img):
    inputs = preprocess_image(face_img)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# Get face encoding
def get_face_encoding(face_img):
    try:
        rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_face_img)
        if not face_locations:
            return None
        face_encodings = face_recognition.face_encodings(rgb_face_img, face_locations)
        return face_encodings[0] if face_encodings else None
    except Exception as e:
        st.error(f"Error in get_face_encoding: {e}")
        return None

# Compare face with known faces
def recognize_face(face_encoding):
    if face_encoding is None:
        return None
    for known_face_id, known_face_encodings in known_faces.items():
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if any(matches):
            return known_face_id
    return None

# Save new face encoding
def save_new_face(face_encoding, face_id, age):
    if face_encoding is not None:
        known_faces[face_id].append(face_encoding)
        face_data[face_id]['age'] = age
    return face_id

# Save persistent data
def save_persistent_data():
    with open(PERSISTENT_FILE, 'wb') as f:
        pickle.dump({
            'known_faces': known_faces,
            'face_data': face_data,
            'face_id': face_id
        }, f)

def main():
    global face_id

    st.title("Face Recognition and Age Estimation")

    # Sidebar for app controls
    st.sidebar.header("Controls")
    run_recognition = st.sidebar.checkbox("Run Face Recognition", value=True)
    save_data = st.sidebar.button("Save Data")

    # Main content
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Error: Could not open video capture.")
        return

    while run_recognition:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        face_locations = face_recognition.face_locations(frame)
        for (top, right, bottom, left) in face_locations:
            face_img = frame[top:bottom, left:right]
            face_encoding = get_face_encoding(face_img)

            if face_encoding is not None:
                known_face = recognize_face(face_encoding)
                if known_face is not None:
                    age = face_data[known_face]['age']
                else:
                    age = predict_age(face_img)
                    face_id = save_new_face(face_encoding, face_id, age)
                    known_face = face_id
                    face_id += 1

                label = f"Person {known_face}: Age {age}"

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the resulting frame
        frame_placeholder.image(frame, channels="BGR")
        info_placeholder.text(f"Number of known faces: {len(known_faces)}")

        if save_data:
            save_persistent_data()
            st.sidebar.success("Data saved successfully!")
            save_data = False

        if not run_recognition:
            break

    video_capture.release()


if __name__ == "__main__":
    main()
