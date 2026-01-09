import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURACIÓN (MODIFICA ESTO)
# ==========================================

# Ruta al archivo .pth que guardaste tras el entrenamiento

MODEL_WEIGHTS_PATH = "./results_seg/mejor_modelo_deeplab_mobilenetv3.pth" 

NUM_CLASSES = 3 # Fondo (0) y Hoja (1)

# El dispositivo debe coincidir con cómo se guardó, pero map_location ayuda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DEFINICIÓN DEL MODELO Y CARGA
# ==========================================

def get_deeplab_model(num_classes):
    # Recreamos la estructura exacta del modelo usado en entrenamiento
    # weights=None porque cargaremos nuestros propios pesos
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


def get_deeplab_model_movilnet(num_classes):
    print("🏗️ Cargando DeepLabV3 con MobileNetV3 Large...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

def load_trained_model(weights_path, device):
    model = get_deeplab_model_movilnet(NUM_CLASSES)
    # map_location es vital si entrenaste en GPU y ahora usas CPU
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    # ¡CRUCIAL! Poner en modo evaluación
    model.eval()
    print(f"Modelo cargado desde: {weights_path}")
    return model

# ==========================================
# 3. UTILIDADES DE PREPROCESAMIENTO Y VISUALIZACIÓN
# ==========================================

# Transformación IDÉNTICA a la usada en validación durante el entrenamiento
inference_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def decode_segmap_to_rgb(mask_np):
    """ Convierte una máscara de índices (0, 1) a una imagen RGB visible """
    # Definimos colores: Fondo=Negro, Clase 1=Verde brillante
    # Formato (R, G, B)
    colors = np.array([
        [0, 0, 0],    # Índice 0 (Fondo)
        [0, 255, 0],  # Índice 1 (Hoja) - Puedes cambiar esto a [255,0,0] para rojo
        [255, 0, 0]
    ]).astype(np.uint8)

    # Crear imagen RGB vacía
    r = np.zeros_like(mask_np).astype(np.uint8)
    g = np.zeros_like(mask_np).astype(np.uint8)
    b = np.zeros_like(mask_np).astype(np.uint8)

    # Asignar colores según el índice de clase
    for label_idx in range(0, NUM_CLASSES):
        idx = (mask_np == label_idx)
        r[idx] = colors[label_idx, 0]
        g[idx] = colors[label_idx, 1]
        b[idx] = colors[label_idx, 2]

    rgb_mask = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb_mask)

def crop_detected_object(original_pil_resized, mask_np, target_class=1):
    """ Recorta el área detectada de la imagen original redimensionada """
    # Encontrar los índices donde la máscara es igual a la clase objetivo (hoja)
    rows, cols = np.where(mask_np == target_class)

    if len(rows) == 0:
        return None # No se detectó nada

    # Calcular bounding box
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)

    # Añadir un pequeño margen (padding) opcional
    margin = 10
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(original_pil_resized.height, y_max + margin)
    x_max = min(original_pil_resized.width, x_max + margin)

    # Realizar el crop sobre la imagen PIL redimensionada (512x512)
    cropped_img = original_pil_resized.crop((x_min, y_min, x_max, y_max))
    return cropped_img

def remove_background(original_pil_resized, mask_np):
    """ 
    Deja transparente todo lo que no sea la clase objetivo.
    Devuelve una imagen RGBA (Red, Green, Blue, Alpha).
    """
    # 1. Convertir la imagen original a un array de NumPy (RGB)
    img_rgb = np.array(original_pil_resized)
    
    # 2. Crear el Canal Alfa (Transparencia)
    # Donde la máscara es igual a la clase (hoja), el alfa es 255 (Opaco).
    # Donde no, el alfa es 0 (Totalmente transparente).
    

    # 3. Limpiar los bordes (Opcional pero recomendado)
    # Pone en negro (0,0,0) los píxeles del fondo en la imagen RGB.
    # Esto evita bordes "sucios" si la transparencia no es perfecta en visualizadores.
    img_rgb[mask_np == 0] = [0, 0, 0]

    # 4. Unir los canales: RGB + Alpha = RGBA
    # np.dstack apila arrays en profundidad (depth)


    # 5. Convertir de vuelta a objeto PIL
    return Image.fromarray(img_rgb)
# ==========================================
# 4. BUCLE PRINCIPAL DE INFERENCIA
# ==========================================

def run_inference(input_folder, output_test_folder, output_final_folder):
    # Preparar directorios
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_final_folder, exist_ok = True)
    save_dir_overlay = os.path.join(output_test_folder, 'overlays')
    save_dir_crop = os.path.join(output_test_folder, 'crops')
    os.makedirs(save_dir_overlay, exist_ok=True)
    os.makedirs(save_dir_crop, exist_ok=True)

    # Cargar modelo
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: No se encuentra el archivo de pesos: {MODEL_WEIGHTS_PATH}")
        print("Por favor, ajusta la ruta en la configuración o entrena el modelo primero.")
        return

    model = load_trained_model(MODEL_WEIGHTS_PATH, DEVICE)

    # Buscar imágenes
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(image_files)
    print(f"{len(image_files)} Imagenes Encontradas")
    if len(image_files) == 0:
        print(f"No se encontraron imágenes en: {input_folder}")
        print("Por favor, coloca algunas imágenes ahí para probar.")
        # Creamos una imagen dummy para que el usuario sepa dónde ponerlas
        dummy = Image.new('RGB', (600, 400), color='gray')
        dummy.save(os.path.join(input_folder, 'imagen_ejemplo_borrame.jpg'))
        print("-> Se ha creado 'imagen_ejemplo_borrame.jpg' como guía.")
        return

    print(f"Iniciando inferencia en {len(image_files)} imágenes usando {DEVICE}...")

    # Desactivar gradientes para inferencia (ahorra memoria y tiempo)
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_folder, img_name)
            
            # 1. Cargar imagen original
            orig_pil = Image.open(img_path).convert("RGB")
            # Guardamos una versión redimensionada para la visualización final
            orig_pil_resized = orig_pil.resize((512, 512))

            # 2. Preprocesar (Transformar a Tensor normalizado)
            # Unsqueeze(0) añade la dimensión del batch: [3, 512, 512] -> [1, 3, 512, 512]
            input_tensor = inference_transform(orig_pil).unsqueeze(0).to(DEVICE)

            # 3. Predicción del modelo
            output = model(input_tensor)['out'] # En inferencia, DeepLab solo devuelve 'out'
            
            # 4. Post-procesamiento
            # Output shape es [1, 2, 512, 512].
            # argmax en dim=1 obtiene el índice de la clase con mayor probabilidad.
            # Resultado shape: [1, 512, 512] -> squeeze -> [512, 512]
            mask_tensor = torch.argmax(output, dim=1).squeeze().cpu()
            mask_np = mask_tensor.numpy() # Máscara final de índices (0 y 1)

            # --- VISUALIZACIÓN 1: OVERLAY (Mezcla) ---
            # Convertir máscara a RGB (verde para la hoja)
            mask_rgb_pil = decode_segmap_to_rgb(mask_np)
            
            # Mezclar imagen original con máscara (alpha=0.5 para 50% de transparencia)
            overlay_img = Image.blend(orig_pil_resized, mask_rgb_pil, alpha=0.5)
            
            overlay_path = os.path.join(save_dir_overlay, img_name)
            overlay_img.save(overlay_path)

            # --- VISUALIZACIÓN 2: CROP (Recorte) ---
            cropped_img = remove_background(orig_pil_resized, mask_np)
            
            if cropped_img is not None:
                crop_test_path = os.path.join(save_dir_crop, img_name)
                cropped_img.save(crop_test_path)
                crop_path = os.path.join(output_final_folder, img_name)
                cropped_img.save(crop_path)
                status = "OK"
            else:
                status = "Sin detección"

            print(f"Procesado: {img_name} -> {status}")

    print(f"\nInferencia terminada. Resultados guardados en: {output_final_folder}")

if __name__ == "__main__":
    # Carpeta donde pondrás las imágenes nuevas para probar
    root_dir_splits = ["./datos/dataset_split/train", "./datos/dataset_split/test"]
    
    for dir_split in root_dir_splits:
        for dir_class in os.listdir(dir_split):
            INPUT_FOLDER = f"{dir_split}/{dir_class}"

        # Carpeta donde se guardarán los resultados
            OUTPUT_FINAL_FOLDER = f"./datos/dataset_split_cropped_pth_v4/{os.path.basename(dir_split)}/{dir_class}"
        
            OUTPUT_TEST_FOLDER = f"./datos/dataset_split_cropped_visualize_pth_v4/{os.path.basename(dir_split)}/{dir_class}"

            run_inference(input_folder=INPUT_FOLDER, output_test_folder = OUTPUT_TEST_FOLDER, output_final_folder = OUTPUT_FINAL_FOLDER)