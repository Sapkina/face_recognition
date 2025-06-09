import streamlit as st
import cv2
import time
import numpy as np
from recognition.face_detection import detection_face
from recognition.face_embedding import get_embedding
from recognition.find_face import find_closest_face
from database.db import load_all_embeddings

# Кеширование базы
@st.cache_data
def get_known_embeddings():
    return load_all_embeddings()

def get_frame_from_camera():
    st.header("Система распознавания лиц")
    frame_placeholder = st.empty()

    known_faces = get_known_embeddings()
    cap = cv2.VideoCapture(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Не удалось получить изображение с камеры.")
                break

            result_detection = detection_face(frame)
            if result_detection:
                bbox, height, width = result_detection
                x1, y2, x2, y1 = bbox
                x1, y1 = int(x1 * width), int(y1 * height)
                x2, y2 = int(x2 * width), int(y2 * height)

                cropped_image = frame[y1:y2, x1:x2]
                frame_embedding = get_embedding(cropped_image)

                result_recognition = find_closest_face(frame_embedding, known_faces)
                if result_recognition:
                    name, similarity, role, group = result_recognition
                    text = f"{name} ({similarity:.2f}) | {role} | {group}"
                    color = (0, 255, 0)
                else:
                    text = "Unknown face"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.05)  # ~20 FPS

    finally:
        cap.release()


