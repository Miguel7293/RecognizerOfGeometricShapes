import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recognizer import Recognizer
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

def crear_interfaz(root):
    root.title("Análisis de Formas Geométricas")
    # Centra y agranda la ventana
    root.geometry("400x300")  # Tamaño de la ventana
    root.eval('tk::PlaceWindow . center')  # Centra la ventana en la pantalla
    
    # Etiqueta de bienvenida
    label = tk.Label(root, text="Selecciona una opción:", font=("Arial", 16))
    label.pack(pady=20)

    # Botón para analizar con cámara (más grande)
    btn_camera = tk.Button(root, text="Analizar con Cámara", command=analizar_con_camara, font=("Arial", 14), width=20, height=2)
    btn_camera.pack(pady=10)

    # Botón para subir imagen (más grande)
    btn_upload = tk.Button(root, text="Subir Imagen", command=subir_imagen, font=("Arial", 14), width=20, height=2)
    btn_upload.pack(pady=10)

def analizar_con_camara():
    recognizer = Recognizer()
    recognizer.run_detection()

def subir_imagen():
    recognizer = Recognizer()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            objects = recognizer.segment_object(image)
            if objects:
                for roi, coords in objects:
                    if roi is not None and coords is not None:
                        forma_detectada, probabilidad = recognizer.detectar_forma(roi)
                        print(f"Forma detectada: {forma_detectada} (Confianza: {probabilidad:.2f}%)")
                        x, y, w, h = coords
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(image, f"{forma_detectada} {probabilidad:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('Imagen Analizada', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No se detectaron objetos en la imagen.")
        else:
            print(f"Error: No se pudo cargar la imagen desde {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    crear_interfaz(root)
    root.mainloop()