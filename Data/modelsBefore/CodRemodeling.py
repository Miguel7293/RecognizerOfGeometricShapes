import sys
import os
sys.path.append(os.path.abspath('..'))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.data_loader import load_data
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import time

# Carga el modelo existente
model_path = os.path.join(os.path.dirname(os.path.abspath('')), 'data', 'models', 'modelo_formas.h5')
model = tf.keras.models.load_model(model_path)
print(f"Modelo cargado desde: {model_path}")

# Carga las clases existentes para mantener consistencia
data_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'data', 'raw')
_, _, forma_to_label = load_data(data_dir, max_images_per_class=0)  # Solo carga las clases
num_classes = len(forma_to_label)
print("Clases originales:", forma_to_label)

# Función para capturar y seleccionar un objeto con cámara
def capture_and_select_object():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None

    print("Cámara activada. Dibuja un rectángulo alrededor del objeto con el mouse y presiona 'c' para capturar. Presiona 'q' para salir.")
    roi = None
    rect = (0, 0, 0, 0)
    drawing = False

    def draw_rectangle(event, x, y, flags, param):
        nonlocal rect, drawing, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect = (rect[0], rect[1], x - rect[0], y - rect[1])
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = (rect[0], rect[1], x - rect[0], y - rect[1])
            roi = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            if roi.size > 0:
                roi = cv2.resize(roi, (128, 128))
                cv2.imshow('ROI Seleccionada', roi)

    cv2.namedWindow('Selecciona Objeto')
    cv2.setMouseCallback('Selecciona Objeto', draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if drawing:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
        cv2.imshow('Selecciona Objeto', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and roi is not None:
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return roi

# Bucle para capturar, verificar y recolectar datos antes de reentrenar
new_images = []
new_labels = []
print("Comenzando reentrenamiento interactivo. Captura objetos con la cámara y verifica las predicciones. Presiona 'q' para salir y reentrenar con todas las imágenes capturadas.")
while True:
    roi = capture_and_select_object()
    if roi is None:
        break

    # Normaliza la ROI
    roi_normalized = roi / 255.0

    # Predice con el modelo
    prediction = model.predict(np.expand_dims(roi_normalized, axis=0), verbose=0)
    clase = np.argmax(prediction)
    probabilidad = np.max(prediction) * 100
    forma_predicha = next((forma for forma, idx in forma_to_label.items() if idx == clase), "Desconocida")
    print(f"Forma predicha: {forma_predicha} (Confianza: {probabilidad:.2f}%)")

    # Pide retroalimentación
    feedback = input("¿Es correcto? (s/n, o 'q' para salir y reentrenar): ").lower()
    if feedback == 'q':
        break
    elif feedback == 's':
        label = clase
        forma_correcta = forma_predicha
    else:
        print("Selecciona la forma correcta (0-23):", forma_to_label)
        try:
            label = int(input("Ingresa el número de la forma correcta: "))
            if label not in range(num_classes):
                print("Número inválido. Ignorando esta imagen.")
                continue
            forma_correcta = next((forma for forma, idx in forma_to_label.items() if idx == label), "Desconocida")
        except ValueError:
            print("Entrada inválida. Ignorando esta imagen.")
            continue

    # Guarda la imagen en la carpeta correspondiente
    save_dir = os.path.join(data_dir, forma_correcta)
    os.makedirs(save_dir, exist_ok=True)
    image_name = f"{forma_correcta}_{len(os.listdir(save_dir)) + 1}.jpg"
    image_path = os.path.join(save_dir, image_name)
    cv2.imwrite(image_path, roi)
    print(f"Imagen guardada en: {image_path}")

    # Añade a los datos nuevos
    new_images.append(roi_normalized)
    new_labels.append(label)

# Si hay nuevas imágenes, reentrena el modelo solo si hay suficientes datos
if new_images and len(new_images) > 1:  # Asegura al menos 2 imágenes
    new_images = np.array(new_images)
    new_labels = tf.keras.utils.to_categorical(new_labels, num_classes)  # Asegura 24 clases

    # Divide los nuevos datos
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(new_images, new_labels, test_size=0.2, random_state=42)

    # Augmentación
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    train_generator = datagen.flow(X_train_new, y_train_new, batch_size=16)
    validation_generator = datagen.flow(X_val_new, y_val_new, batch_size=16)

    # Recompila si es necesario
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Reentrena el modelo
    print("Reentrenando modelo con nuevas imágenes...")
    model.fit(train_generator, epochs=5, validation_data=validation_generator)

    # Guarda el modelo actualizado
    model.save(model_path)
    print(f"Modelo reentrenado y guardado en: {model_path}")
else:
    print("No se capturaron suficientes imágenes para reentrenar. Se necesitan al menos 2 imágenes.")