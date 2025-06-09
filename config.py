import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PHOTO_DIR = os.path.join(PROJECT_ROOT, 'database', 'photo')
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'models', 'detector_weights.pth')
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'face_recognition.db')

print(DB_PATH)