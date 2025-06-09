import streamlit as st
from tabs.camera_tab import get_frame_from_camera
from tabs.add_user_tab import add_user_to_system

def main():
    st.set_page_config(page_title="Face Recognition App", layout="wide")

    # Выбор вкладки через selectbox
    page = st.selectbox("Выберите страницу:", ["Камера", "Добавить пользователя"])

    if page == "Камера":
        # Потоковое видео с камеры
        video_feed = st.empty()  # Пустое место для видео
        for frame in get_frame_from_camera():
            video_feed.image(frame, channels="BGR", use_column_width=True)

    elif page == "Добавить пользователя":
        with st.container():  # Контейнер, чтобы ограничить область отображения формы
            add_user_to_system()

if __name__ == "__main__":
    main()





