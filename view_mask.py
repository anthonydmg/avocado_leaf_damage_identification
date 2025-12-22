import numpy as np
import matplotlib.pyplot as plt

def visualizar_mascara_npy(ruta_archivo):
    # 1. Cargar el archivo .npy
    try:
        mascara = np.load(ruta_archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo}")
        return

    # Asegurarnos de que la máscara sea 2D (Alto, Ancho)
    # Si viene como (1, Alto, Ancho) o (Alto, Ancho, 1), esto lo aplana
    mascara = np.squeeze(mascara)

    # 2. Crear una imagen RGB vacía (negra por defecto)
    # Dimensiones: (Alto, Ancho, 3 canales de color)
    alto, ancho = mascara.shape
    imagen_visual = np.zeros((alto, ancho, 3), dtype=np.uint8)

    # Nota: np.zeros ya inicializa todo en [0, 0, 0] (Negro),
    # por lo que el fondo (valor 0) ya está listo.

    # 3. Asignar colores según la clase
    # Formato RGB: [Rojo, Verde, Azul]
    
    # Clase 1: Azul -> [0, 0, 255]
    imagen_visual[mascara == 1] = [0, 0, 255]
    
    # Clase 2: Rojo -> [255, 0, 0]
    imagen_visual[mascara == 2] = [255, 0, 0]

    # 4. Visualizar
    plt.figure(figsize=(8, 8))
    plt.title("Visualización de Máscara (0=Negro, 1=Azul, 2=Rojo)")
    plt.imshow(imagen_visual)
    plt.axis('off') # Ocultar ejes para ver solo la imagen
    plt.show()

# --- Ejemplo de uso ---
visualizar_mascara_npy('./datos/segmentacion/etiquetados/masks/masks_long/20251021_091321_MASK_LONG.npy')