import streamlit as st
import sqlite3
import pickle
from PIL import Image
import numpy as np
from recognition.face_embedding import get_embedding
from config import DB_PATH

def add_user_to_system():
    st.header("Добавить нового пользователя в базу")

    # Поля для ввода информации о пользователе
    name = st.text_input("Имя:")
    role = st.text_input("Роль:")
    group = st.text_input("Номер группы:")
    uploaded_file = st.file_uploader("Загрузить фото:", type=["jpg", "jpeg", "png"])

    if uploaded_file and name and role and group:
        # Считываем изображение
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)  # Конвертируем в NumPy массив

        # Преобразуем изображение в формат BGR (OpenCV)
        image_bgr = np.array(image)  # Не используем cv2, так как оно не нужно для этого

        # Получаем эмбеддинг изображения
        embedding = get_embedding(image_bgr)
        db_embedding = pickle.dumps(embedding)

        # Кнопка для добавления в базу
        if st.button("Добавить в базу"):
            # Сериализация изображения в бинарный формат
            image_bytes = uploaded_file.read()

            # Добавляем данные в базу
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO university (name, embedding, role, group_name, photo)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, db_embedding, role, group, image_bytes))  # Добавляем фото как бинарные данные
            conn.commit()
            conn.close()

            # Очистка кэша
            st.cache_data.clear()

            # Уведомление о добавлении
            st.success(f"Пользователь {name} успешно добавлен!")

    elif uploaded_file and (not name or not role or not group):
        st.warning("Пожалуйста, заполните все поля перед добавлением.")




