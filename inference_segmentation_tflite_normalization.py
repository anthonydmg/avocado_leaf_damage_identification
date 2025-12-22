import numpy as np
import os
from PIL import Image
import tensorflow as tf # O 'import tflite_runtime.interpreter as tflite' para dispositivos embebidos

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

# Ruta al archivo .tflite
MODEL_TFLITE_PATH = "./deeplab_v3_leaf.tflite" 

NUM_CLASSES = 3 
INPUT_SIZE = (512, 512)

# Parámetros de normalización (deben ser los mismos que en entrenamiento)
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

# ==========================================
# 2. CARGA DEL MODELO TFLITE
# ==========================================

def load_tflite_model(model_path):
    # Cargar el intérprete y asignar tensores
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Obtener detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Modelo TFLite cargado: {model_path}")
    return interpreter, input_details, output_details

# ==========================================
# 3. PREPROCESAMIENTO Y UTILIDADES
# ==========================================

def preprocess_image(pil_img, size, mean, std):
    """ Replica transforms.Compose de PyTorch """
    # 1. Resize
    img = pil_img.resize(size, Image.BILINEAR)
    # 2. ToTensor (Escalar a [0, 1])
    img_array = np.array(img).astype(np.float32) / 255.0
    # 3. Normalize
    img_array = (img_array - mean) / std
    # 4. Add Batch Dimension [1, H, W, C]
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def decode_segmap_to_rgb(mask_np):
    colors = np.array([
        [0, 0, 0],     # Fondo
        [0, 255, 0],   # Clase 1 (Hoja)
        [255, 0, 0]    # Clase 2 (Enfermedad/Otro)
    ]).astype(np.uint8)

    rgb_mask = colors[mask_np] # Indexación avanzada de NumPy para mayor velocidad
    return Image.fromarray(rgb_mask)

def remove_background(original_pil_resized, mask_np):
    img_rgb = np.array(original_pil_resized)
    # Aplicar máscara: lo que no es hoja (o clase de interés) se vuelve negro
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

    if not os.path.exists(MODEL_TFLITE_PATH):
        print(f"Error: No existe {MODEL_TFLITE_PATH}")
        return

    # Cargar intérprete
    interpreter, input_details, output_details = load_tflite_model(MODEL_TFLITE_PATH)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        orig_pil = Image.open(img_path).convert("RGB")
        orig_pil_resized = orig_pil.resize(INPUT_SIZE)

        # 1. Preprocesar
        input_data = preprocess_image(orig_pil, INPUT_SIZE, NORM_MEAN, NORM_STD)

        # 2. Ejecutar Inferencia
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 3. Obtener resultado
        # Nota: La forma de salida de TFLite suele ser [1, H, W, C] o [1, C, H, W]
        # Depende de cómo exportaste el modelo. Ajustamos si es necesario.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Si la salida es [1, C, H, W], movemos los ejes a [1, H, W, C]
        if output_data.shape[1] == NUM_CLASSES:
            output_data = np.transpose(output_data, (0, 2, 3, 1))

        # 4. Post-procesamiento (Argmax en el eje de las clases)
        mask_np = np.argmax(output_data[0], axis=-1).astype(np.uint8)

        # --- Visualización ---
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
            
            path_out_final = f"./datos/dataset_split_cropped/{os.path.basename(dir_split)}/{dir_class}"
            path_out_viz = f"./datos/dataset_split_cropped_visualize/{os.path.basename(dir_split)}/{dir_class}"

            run_inference(path_in, path_out_viz, path_out_final)