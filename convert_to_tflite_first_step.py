import torch
import torch.nn as nn
from torchvision import models

import onnx
import numpy as np


# ================================================================
# 1. CARGAR TU MODELO (igual al que usaste entrenando)
# ================================================================
def load_model(num_classes, device):
    print("Cargando modelo EfficientNetV2-S pre-entrenado...")

    weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_m(weights=weights)

    # Congelar capas base
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar clasificador
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


# ================================================================
# 2. EXPORTAR A ONNX
# ================================================================
def export_to_onnxprev(model, save_path="model.onnx", input_size=(1,3,224,224)):
    print("\n📤 Exportando a ONNX...")
    model.eval()

    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17
    )

    print(f"✅ Modelo exportado a ONNX: {save_path}")


# ================================================================
# 3. ONNX → TENSORFLOW


def export_to_onnx(model, export_path="model.onnx"):
    model_cpu = model.to("cpu")
    model_cpu.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,          # recomendado para compatibilidad
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        },
        dynamo=False               # ¡esto hace que PyTorch 2.9 use el exportador viejo!
    )

    print(f"✔ ONNX guardado en: {export_path}")

# ================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================
if __name__ == "__main__":

    # ---- AJUSTAR ESTO ----
    num_classes = 6  # Cambia según tu dataset
    model_path = "./efficientnetv2_m_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar modelo base
    model = load_model(num_classes, device)

    # 2. Cargar tus pesos entrenados
    print("\n📥 Cargando pesos entrenados...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("   ✔ Pesos cargados correctamente.")

    # 3. Exportar a ONNX
    export_to_onnx(model, export_path= "./model.onnx")


    print("\n🚀 Conversión completa con éxito.\n")
