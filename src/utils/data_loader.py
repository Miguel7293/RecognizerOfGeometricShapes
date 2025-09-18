import os
import cv2
import numpy as np

def load_data(data_dir, max_images_per_class=100):
    images = []
    labels = []
    formas = sorted(os.listdir(data_dir))
    if not formas:
        raise ValueError("No se encontraron carpetas en data_dir.")
    
    forma_to_label = {forma: label for label, forma in enumerate(formas)}
    
    # Si max_images_per_class=0, solo devuelve las clases sin cargar imágenes
    if max_images_per_class == 0:
        return np.array([]), np.array([]), forma_to_label
    
    for forma in formas:
        path = os.path.join(data_dir, forma)
        if not os.path.isdir(path):
            print(f"Advertencia: {path} no es una carpeta. Saltando.")
            continue
        label = forma_to_label[forma]
        img_count = 0
        max_limit = float('inf') if max_images_per_class is None else max_images_per_class
        for img_name in os.listdir(path):
            if img_count >= max_limit:
                break
            img_path = os.path.join(path, img_name)
            if not os.path.isfile(img_path):
                print(f"Advertencia: {img_path} no es un archivo. Saltando.")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Advertencia: No se pudo cargar {img_path}. Saltando.")
                continue
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
            img_count += 1
    if not images and max_images_per_class != 0:
        raise ValueError("No se cargaron imágenes. Verifica data_dir y los archivos.")
    return np.array(images), np.array(labels), forma_to_label