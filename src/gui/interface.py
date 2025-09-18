from src.models.recognizer import Recognizer
import tkinter as tk

def crear_interfaz(root):
    root.title("Reconocedor de Formas 2D y 3D")
    btn = tk.Button(root, text="Iniciar Detección", command=analizar)
    btn.pack(pady=20)

def analizar():
    recognizer = Recognizer()
    recognizer.detectar_forma()

if __name__ == "__main__":
    root = tk.Tk()
    crear_interfaz(root)
    root.mainloop()