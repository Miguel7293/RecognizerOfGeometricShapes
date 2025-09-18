import cv2
import tensorflow as tf
import numpy as np
import os

class Recognizer:
    def __init__(self):
        # Ruta del único modelo
        base_path = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_path, '..', 'data', 'models', 'modelo_formas.h5')
        print(f"Cargando modelo desde: {model_path}")
        
        # Carga el modelo
        self.model = tf.keras.models.load_model(model_path)
        self.formas = sorted(['circle', 'cone', 'cube', 'cuboid', 'cylinder', 'decagon', 'dodecahedron', 'ellipse', 'heptagon', 'hexagon', 
                              'icosahedron', 'nonagon', 'octagon', 'octahedron', 'parallelogram', 'pentagon', 'prism', 'pyramid', 
                              'rectangle', 'rhombus', 'sphere', 'square', 'tetrahedron', 'trapezoid', 'triangle'])

    def segment_object(self, frame):
        # Segmentación básica con contornos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    return cv2.resize(roi, (128, 128)), (x, y, w, h)  # Devuelve ROI y coordenadas
        return None, None

    def detectar_forma(self, image):
        if image is None:
            return "Sin objeto detectable", 0.0
        processed_img = cv2.resize(image, (128, 128))
        processed_img = processed_img / 255.0
        prediction = self.model.predict(np.expand_dims(processed_img, axis=0), verbose=0)
        clase = np.argmax(prediction)
        probabilidad = np.max(prediction) * 100  # Porcentaje de confianza
        forma_detectada = self.formas[clase] if clase < len(self.formas) else "Desconocida"
        return forma_detectada, probabilidad

    def analyze_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path}")
            return "Error", 0.0
        roi, coords = self.segment_object(image)
        return self.detectar_forma(roi if roi is not None else image)

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        print("Cámara activada. Analizando en cada frame. Presiona 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar frame.")
                break

            # Segmenta y analiza el objeto más grande en cada frame
            roi, coords = self.segment_object(frame)
            if roi is not None and coords is not None:
                forma_detectada, probabilidad = self.detectar_forma(roi)
                print(f"Forma detectada: {forma_detectada} (Confianza: {probabilidad:.2f}%)")
                x, y, w, h = coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{forma_detectada} {probabilidad:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Detección de Formas', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

from src.utils.image_utils import preprocess_image