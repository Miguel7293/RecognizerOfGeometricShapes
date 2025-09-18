import cv2
import tensorflow as tf
import numpy as np
import os

class Recognizer:
    def __init__(self):
        # Ruta del modelo
        base_path = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_path, '..', 'data', 'models', 'modelo_formas.h5')
        print(f"Cargando modelo desde: {model_path}")
        
        # Carga el modelo
        self.model = tf.keras.models.load_model(model_path)
        # Define las clases seleccionadas (debe coincidir con exploracion.ipynb)
        self.formas = ['circle', 'cone', 'cube', 'cuboid', 'cylinder', 'ellipse', 'hexagon', 'octagon', 'pentagon', 'prism', 'pyramid', 'rectangle', 'rhombus', 'sphere', 'square', 'triangle']

    def segment_object(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)  # Umbral más bajo para detectar más objetos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Umbral más bajo (de 1000 a 500) para detectar objetos más pequeños
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0 and w > 20 and h > 20:  # Verifica que la ROI sea razonable
                        roi_resized = cv2.resize(roi, (128, 128))
                        objects.append((roi_resized, (x, y, w, h)))
        return objects if objects else []

    def detectar_forma(self, image):
        if image is None:
            return "Sin objeto detectable", 0.0
        processed_img = cv2.resize(image, (128, 128))
        processed_img = processed_img / 255.0
        prediction = self.model.predict(np.expand_dims(processed_img, axis=0), verbose=0)
        clase = np.argmax(prediction)
        probabilidad = np.max(prediction) * 100
        forma_detectada = self.formas[clase] if clase < len(self.formas) else "Desconocida"
        return forma_detectada, probabilidad

    def analyze_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path}")
            return "Error", 0.0
        objects = self.segment_object(image)
        results = []
        for roi, _ in objects:
            if roi is not None:
                results.append(self.detectar_forma(roi))
        return results[0] if results else ("Sin objeto detectable", 0.0)

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        print("Cámara activada. Analizando múltiples objetos en cada frame. Presiona 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar frame.")
                break

            # Segmenta y analiza todos los objetos en cada frame (análisis continuo)
            objects = self.segment_object(frame)
            if objects:
                print(f"Detectados {len(objects)} objetos.")
                for roi, coords in objects:
                    if roi is not None and coords is not None:
                        forma_detectada, probabilidad = self.detectar_forma(roi)
                        x, y, w, h = coords
                        # Dibuja el cuadrado verde alrededor del objeto
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        # Muestra la figura y el porcentaje en el cuadro
                        cv2.putText(frame, f"{forma_detectada} {probabilidad:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        print(f"Objeto en ({x}, {y}): {forma_detectada} ({probabilidad:.2f}%)")
            else:
                print("No se detectaron objetos en este frame.")

            cv2.imshow('Detección de Formas', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

from src.utils.image_utils import preprocess_image