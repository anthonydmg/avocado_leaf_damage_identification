import torch
import torch.nn as nn
from torchvision import models
import ai_edge_torch
import numpy as np
import os

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
class DeepLabExportWrapper(nn.Module):
    """
    Clase envoltorio para limpiar la salida.
    DeepLab devuelve {'out': ..., 'aux': ...}.
    Para TFLite solo queremos el tensor 'out'.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Ejecutar modelo
        output = self.model(x)
        # Devolver SOLO la predicción principal (ignoramos 'aux')
        return output['out']

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
    wrapped_model = DeepLabExportWrapper(base_model)
    wrapped_model.to(device)
    wrapped_model.eval()
    
    return wrapped_model

if __name__ == "__main__":
    # ---- CONFIGURACIÓN ----
    NUM_CLASSES = 3  # Ajusta a tus clases (0=Fondo, 1=Hoja)
    IMG_SIZE = 512   # El tamaño usado en el entrenamiento
    MODEL_PATH = "./results_seg/mejor_modelo_deeplab_mobilenetv3.pth" # Tu archivo .pth
    EXPORT_PATH = "./results_seg/deeplab_v3_mobilenetlarge.tflite"
    
    DEVICE = "cpu" # Para exportar suele ser mejor usar CPU para evitar conflictos de memoria

    # 1. Cargar el modelo listo para exportar
    try:
        model = load_trained_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo .pth. Verifica la ruta.")
        exit()
    
    model.eval()
    print("\n🔄 Preparando conversión a TFLite (AI Edge)...")

    # 2. Convertir a formato 'Channels Last' (NHWC)
    # Esto es vital para que corra rápido en celulares (Android/NPU).
    # args=[0] indica que el primer argumento (la imagen) debe transformarse.
    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0])

    # 3. Crear input de ejemplo en formato NHWC (Batch, Alto, Ancho, Canales)
    # PyTorch usa NCHW, pero ai_edge_torch espera el input "como lo vería el celular"
    sample_input = (torch.randn(1, IMG_SIZE, IMG_SIZE, 3),)

    print("🚀 Convirtiendo... (Esto puede tardar un poco)")
    
    # 4. Convertir y Optimizar
    edge_model = ai_edge_torch.convert(nhwc_model, sample_input)

    # 5. Verificación numérica (Opcional pero recomendada)
    print("🔍 Verificando consistencia numérica...")
    # Ejecutamos el modelo original (envuelto en NHWC)
    with torch.no_grad():
        torch_out = nhwc_model(*sample_input) # Salida Tensor PyTorch
    
    # Ejecutamos el modelo convertido
    edge_out = edge_model(*sample_input)   # Salida NumPy array
    
    # Nota: edge_out puede venir como tupla, extraemos el primer elemento si es necesario
    if isinstance(edge_out, (list, tuple)):
        edge_out = edge_out[0]

    # Comparamos
    torch_out_np = torch_out.detach().numpy()
    
    # DeepLab es complejo, toleramos un error pequeño (1e-3)
    is_close = np.allclose(torch_out_np, edge_out, atol=1e-3, rtol=1e-3)
    print(f"   Coincidencia de salidas: {'✅ ÉXITO' if is_close else '⚠️ DIVERGENCIA (Revisar)'}")

    # 6. Guardar archivo final
    edge_model.export(EXPORT_PATH)
    print(f"\n💾 Modelo guardado exitosamente en: {EXPORT_PATH}")
