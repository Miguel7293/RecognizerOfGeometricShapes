import cv2
import numpy as np

def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0  # Normalización
    return img