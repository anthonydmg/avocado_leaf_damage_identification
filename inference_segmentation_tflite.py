import numpy as np
import os
from PIL import Image
import tensorflow as tf # O 'import tflite_runtime.interpreter as tflite' para dispositivos embebidos

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

MODEL_TFLITE_PATH = "./deeplab_paltas_calibrado_int8.tflite" 
NUM_CLASSES = 3 
INPUT_SIZE = (512, 512)

# YA NO NECESITAS NORM_MEAN NI NORM_STD AQUÍ 
# El modelo se encarga internamente.

# ==========================================
# 2. CARGA DEL MODELO TFLITE
# ==========================================

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Modelo TFLite cargado: {model_path}")
    return interpreter, input_details, output_details

# ==========================================
# 3. PREPROCESAMIENTO SIMPLIFICADO
# ==========================================

def preprocess_image(pil_img, size):
    """ 
    Preprocesamiento mínimo porque la normalización está DENTRO del TFLite.
    """
    # 1. Redimensionar al tamaño que espera el modelo
    img = pil_img.resize(size, Image.BILINEAR)
    
    # 2. Convertir a Array de Numpy (Valores 0-255)
    # Importante: El modelo en el wrapper espera uint8 o float32 sin normalizar
    img_array = np.array(img).astype(np.float32) 
    
    # 3. Añadir dimensión de Batch [1, 512, 512, 3]
    return np.expand_dims(img_array, axis=0)

def decode_segmap_to_rgb(mask_np):
    colors = np.array([
        [0, 0, 0],     # Fondo
        [0, 255, 0],   # Clase 1 (Hoja)
        [255, 0, 0]    # Clase 2
    ]).astype(np.uint8)
    rgb_mask = colors[mask_np]
    return Image.fromarray(rgb_mask)

def remove_background(original_pil_resized, mask_np):
    img_rgb = np.array(original_pil_resized)
    img_rgb[mask_np == 0] = [0, 0, 0]
    return Image.fromarray(img_rgb)

# ==========================================
# 4. BUCLE DE INFERENCIA
# ==========================================

def run_inference(input_folder, output_test_folder, output_final_folder):
    os.makedirs(output_final_folder, exist_ok=True)
    save_dir_overlay = os.path.join(output_test_folder, 'overlays')
    save_dir_crop = os.path.join(output_test_folder, 'crops')
    os.makedirs(save_dir_overlay, exist_ok=True)
    os.makedirs(save_dir_crop, exist_ok=True)

    interpreter, input_details, output_details = load_tflite_model(MODEL_TFLITE_PATH)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        orig_pil = Image.open(img_path).convert("RGB")
        orig_pil_resized = orig_pil.resize(INPUT_SIZE)

        # 1. Preprocesar (Mucho más rápido y simple)
        input_data = preprocess_image(orig_pil, INPUT_SIZE)

        # 2. Ejecutar Inferencia
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 3. Obtener resultado
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Como usamos el Wrapper, la salida suele venir en NCHW de PyTorch [1, 3, 512, 512]
        # Si es así, la corregimos a NHWC para el argmax
        if output_data.shape[1] == NUM_CLASSES:
            output_data = np.transpose(output_data, (0, 2, 3, 1))

        # 4. Crear máscara
        mask_np = np.argmax(output_data[0], axis=-1).astype(np.uint8)

        # --- Guardar resultados ---
        mask_rgb_pil = decode_segmap_to_rgb(mask_np)
        overlay_img = Image.blend(orig_pil_resized, mask_rgb_pil, alpha=0.5)
        overlay_img.save(os.path.join(save_dir_overlay, img_name))

        cropped_img = remove_background(orig_pil_resized, mask_np)
        if cropped_img:
            cropped_img.save(os.path.join(output_final_folder, img_name))
            cropped_img.save(os.path.join(save_dir_crop, img_name))

        print(f"Procesado: {img_name}")

if __name__ == "__main__":
    # Ajusta estas rutas a tu estructura de carpetas
    dir_splits = ["./datos/dataset_split/train", "./datos/dataset_split/test"]
    
    for dir_split in dir_splits:
        if not os.path.exists(dir_split): continue
        for dir_class in os.listdir(dir_split):
            path_in = os.path.join(dir_split, dir_class)
            if not os.path.isdir(path_in): continue
            
            path_out_final = f"./datos/dataset_split_cropped_tflite/{os.path.basename(dir_split)}/{dir_class}"
            path_out_viz = f"./datos/dataset_split_cropped_visualize_tflite/{os.path.basename(dir_split)}/{dir_class}"

            run_inference(path_in, path_out_viz, path_out_final)