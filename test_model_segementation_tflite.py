import numpy as np
import tensorflow as tf # O import tflite_runtime.interpreter as tflite
import cv2
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_PATH = "deeplab_v3_leaf.tflite"
IMAGE_PATH = "hoja_test.jpeg"  # Pon aquí una imagen real tuya
IMG_SIZE = 512
NUM_CLASSES = 3 

# Colores para visualizar (R, G, B) para cada clase
# 0: Fondo (Negro), 1: Hoja (Verde), 2: Daño/Otro (Rojo)
CLASS_COLORS = np.array([
    [0, 0, 0],       # Clase 0
    [0, 255, 0],     # Clase 1
    [255, 0, 0]      # Clase 2
], dtype=np.uint8)

# ==========================================
# 1. CARGAR EL INTÉRPRETE TFLITE
# ==========================================
print(f"Cargando modelo: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Verificar qué forma espera el modelo
input_shape = input_details[0]['shape']
print(f"El modelo espera entrada con forma: {input_shape}") 
# Debería imprimir [1, 512, 512, 3] gracias a to_channel_last_io

# ==========================================
# 2. PREPROCESAMIENTO DE IMAGEN
# ==========================================
def preprocess_image(img_path, target_size):
    # a. Leer imagen con OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    
    # b. Convertir BGR a RGB (OpenCV lee en BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # c. Redimensionar
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # d. Normalización (IGUAL QUE EN TU ENTRENAMIENTO PYTORCH)
    # PyTorch standard mean/std para modelos pre-entrenados
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_normalized = (img_float - mean) / std
    
    # e. Añadir dimensión de Batch: (512, 512, 3) -> (1, 512, 512, 3)
    # NOTA: Como usaste to_channel_last_io, NO hacemos transposición a (C,H,W).
    # Se queda en (H,W,C) que es lo que espera TFLite.
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # f. Asegurar tipo float32
    return img_input.astype(np.float32), img_resized

try:
    input_data, original_img = preprocess_image(IMAGE_PATH, IMG_SIZE)
except FileNotFoundError as e:
    print(e)
    exit()

# ==========================================
# 3. EJECUTAR INFERENCIA
# ==========================================
print("Ejecutando inferencia...")

# Asignar tensor de entrada
interpreter.set_tensor(input_details[0]['index'], input_data)

# Correr el modelo
interpreter.invoke()

# Obtener tensor de salida
output_data = interpreter.get_tensor(output_details[0]['index'])

# output_data tendrá forma [1, 512, 512, NUM_CLASSES] (por el channel_last)
# o [1, NUM_CLASSES, 512, 512] dependiendo de cómo lo exportó exactamente AI Edge.
# Vamos a inspeccionarlo:
print(f"Forma de salida cruda: {output_data.shape}")

# ==========================================
# 4. POST-PROCESAMIENTO
# ==========================================
# Si la salida es [1, 512, 512, 3], usamos argmax en el último eje (axis=-1)
# Si fuera [1, 3, 512, 512], usaríamos axis=1.
if output_data.shape[-1] == NUM_CLASSES:
    prediction_mask = np.argmax(output_data, axis=-1) # [1, 512, 512]
else:
    prediction_mask = np.argmax(output_data, axis=1)  # [1, 512, 512]

# Quitar dimensión batch para visualizar
prediction_mask = prediction_mask[0] # Ahora es [512, 512]

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================
# Crear una imagen coloreada basada en la máscara
segmentation_overlay = CLASS_COLORS[prediction_mask]

# Mostrar resultados
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(original_img)
ax[0].set_title("Imagen Original")
ax[0].axis('off')

ax[1].imshow(prediction_mask, cmap='gray')
ax[1].set_title("Máscara (Clases)")
ax[1].axis('off')

ax[2].imshow(original_img)
ax[2].imshow(segmentation_overlay, alpha=0.5) # Superponer con transparencia
ax[2].set_title("Superposición")
ax[2].axis('off')

plt.tight_layout()
plt.show()

print("✅ Prueba finalizada.")