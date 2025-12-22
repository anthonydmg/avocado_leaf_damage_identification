import torch
import torch.nn as nn
from torchvision import models
import ai_edge_torch

import numpy as np


# ================================================================
# 1. CARGAR TU MODELO (igual al que usaste entrenando)
# ================================================================
def load_model(num_classes, device):
    print("Cargando modelo EfficientNetV2-S pre-entrenado...")

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)

    # Congelar capas base
    #for param in model.parameters():
    #    param.requires_grad = False

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

if __name__ == "__main__":

    # ---- AJUSTAR ESTO ----
    num_classes = 6  # Cambia según tu dataset
    model_path = "./models/torch-models/efficientnetv2_s_final.pth"

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar modelo base
    model = load_model(num_classes, device)

    # 2. Cargar tus pesos entrenados
    print("\n📥 Cargando pesos entrenados...")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("   ✔ Pesos cargados correctamente.")
    model.eval()

    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0])
    
    sample_input = (torch.randn(1, 224, 224, 3),)
    
    output_torch = nhwc_model(*sample_input)
    
    edge_model = ai_edge_torch.convert(nhwc_model, sample_input)
    
    output_edge = edge_model(*sample_input)
    
    
    torch_out_np = output_torch.detach().numpy()
    edge_out_np = output_edge  # ya debería ser numpy
    print(np.allclose(torch_out_np, edge_out_np, atol=1e-5, rtol=1e-5))
    edge_model.export("./models/tflite/efficientnetv2_s_final.tflite")


