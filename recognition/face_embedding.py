import cv2
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1


embedder = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_emb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_image_tensor = transform(image).unsqueeze(0)
    return input_image_tensor

def get_embedding(image):
    input_tensor = preprocess_emb(image)
    embedding = embedder(input_tensor)
    embedding = embedding.detach().cpu().numpy().ravel().tolist()

    return embedding
