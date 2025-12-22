import torch
import torch.nn as nn
from torchvision import models
import ai_edge_torch
import numpy as np
import os
# Importaciones específicas según la arquitectura de la librería
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import cv2
import random
from tqdm import tqdm
# ================================================================
# 1. DEFINICIÓN DEL MODELO (Igual al entrenamiento)
# ================================================================
def get_deeplab_model(num_classes):
    print("🏗️ Cargando DeepLabV3 con MobileNetV3 Large...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

# ================================================================
# 2. WRAPPER (CRÍTICO PARA SEGMENTACIÓN)
# ================================================================
class DeepLabTFLiteWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Registramos las constantes de normalización de ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x entra como (1, 512, 512, 3) y valores 0-255 (NHWC)
        # 1. Pasar a (1, 3, 512, 512) (NCHW)
        x = x.permute(0, 3, 1, 2)
        # 2. Escalar a [0, 1] y Normalizar
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        
        # 3. Inferencia
        output = self.model(x)
        return output['out'] # Salida (1, 3, 512, 512)

# ================================================================
# 3. CARGAR PESOS
# ================================================================
def load_trained_model(model_path, num_classes, device):
    print(f"Construyendo modelo DeepLabV3 para {num_classes} clases...")
    base_model = get_deeplab_model(num_classes)
    
    print(f"Cargando pesos desde: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    base_model.load_state_dict(checkpoint)
    
    # Envolvemos el modelo para que devuelva tensores limpios
    wrapped_model = DeepLabTFLiteWrapper(base_model)
    wrapped_model.eval()
    wrapped_model.to(device)
    
    return wrapped_model

if __name__ == "__main__":

    # 1. Configuración de rutas y parámetros
    MODEL_PATH = "mejor_modelo_deeplab_mobilenatv3.pth"
    RUTA_IMAGENES_PALTAS = "./datos/segmentacion/all_etiquetados/images"  # Carpeta con tus fotos reales
    IMG_SIZE = 512
    NUM_CLASSES = 3
    
    # 2. Cargar el modelo envuelto (Wrapper)
    # Es vital cargarlo en CPU para la exportación
    model = load_trained_model(MODEL_PATH, NUM_CLASSES, "cpu")
    model.eval()
    
    # Input de ejemplo (Batch, Alto, Ancho, Canales) - NHWC
    sample_args = (torch.randn(1, IMG_SIZE, IMG_SIZE, 3),)

    # 3. Paso de Exportación (Nativo de PyTorch 2.9)
    # Esto convierte tu código Python en un grafo de operaciones puro
    exported_model = torch.export.export(model, sample_args).module()

    # 4. Configurar el Cuantizador para Calibración Estática
    # IMPORTANTE: is_dynamic=False para que use los datos de las imágenes
    pt2e_quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )

    # 5. Preparar el modelo para observación
    # Aquí se insertan los "observadores" que medirán los rangos de tus paltas
    prepared_model = prepare_pt2e(exported_model, pt2e_quantizer)

    # 6. Bucle de Calibración (Representative Dataset)
    print("📸 Calibrando rangos con imágenes reales de hojas...")
    lista_fotos = [os.path.join(RUTA_IMAGENES_PALTAS, f) for f in os.listdir(RUTA_IMAGENES_PALTAS) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    seleccion_aleatoria_imagenes = random.sample(lista_fotos, 100)

    with torch.no_grad():
        for path in tqdm(seleccion_aleatoria_imagenes, desc="Preparando:"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Crear el tensor NHWC (1, 512, 512, 3) tal como lo espera el Wrapper
            input_tensor = torch.from_numpy(img).unsqueeze(0).float()
            
            # Pasar la imagen por el modelo preparado para registrar los valores
            prepared_model(input_tensor)

    # 7. Convertir a modelo cuantizado (Weights + Activations a INT8)
    # fold_quantize=False es la recomendación para AI Edge Torch
    quantized_model = convert_pt2e(prepared_model, fold_quantize=False)

    # 8. Conversión Final a TFLite
    print("🚀 Generando archivo TFLite optimizado...")
    edge_model = ai_edge_torch.convert(
        quantized_model,
        sample_args,
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer)
    )

    # 9. Guardar
    edge_model.export("deeplab_paltas_calibrado_int8.tflite")
    print("✅ Proceso terminado. El modelo ya tiene los rangos optimizados.")