import torch
from models.face_detector import FaceDetectionModel
from config import WEIGHTS_PATH
import cv2
from PIL import Image
import torchvision.transforms as transforms

detector = FaceDetectionModel()
detector.load_state_dict(torch.load(WEIGHTS_PATH))
detector.eval()

def preprocess_image(image):
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image_tensor = transform(image).unsqueeze(0)
    return input_image_tensor, height, width

def detection_face(image):
    input_tensor, height, width = preprocess_image(image)

    with torch.no_grad():
        label, bbox = detector(input_tensor)

    label = label.item()
    bbox = bbox.numpy().flatten()

    if label > 0.5:
        return bbox, height, width
    return None




