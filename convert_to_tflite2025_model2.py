import torch
import torch.nn as nn
from torchvision import models
import ai_edge_torch
import timm
import numpy as np


# ================================================================
# 1. CARGAR TU MODELO (igual al que usaste entrenando)
# ================================================================
def load_model(num_classes, device, timm_model_name="mobilevitv2_100.cvnets_in1k", pretrained=True):
    """
    Carga MobileViT-V2 desde timm, reemplaza el clasificador y congela el backbone.
    timm_model_name: nombre del modelo en timm (hay varias variantes: _050, _075, _100, _200, etc.)
    """
    print(f"Cargando modelo {timm_model_name} desde timm (pretrained={pretrained})...")

    # Crear el modelo con timm (si num_classes se pasa, timm crea una nueva cabeza)
    model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=num_classes)
    print("model:", model)
    print()
    # --- Congelar todo primero ---
    for param in model.parameters():
        param.requires_grad = False

    ## Descongelar ultima capa
    # --- Asegurar que la cabeza/classifier esté entrenable ---
    # timm proporciona utilidades: get_classifier(), reset_classifier(), but classifier varies by model.

    model = model.to(device)
    print("Modelo listo (MobileViT-V2). Solo la(s) última(s) capa(s) están entrenables por defecto.")
    print("-" * 30)
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
    model_path = "./models/torch-models/mobilevit_s_final.pth"

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar modelo base
    model = load_model(num_classes, device)

    # 2. Cargar tus pesos entrenados
    print("\n📥 Cargando pesos entrenados...")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("   ✔ Pesos cargados correctamente.")
    model.eval()

    #nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0])
    
    sample_input = (torch.randn(1, 3, 224, 224),)
    
    output_torch = model(*sample_input)
    
    edge_model = ai_edge_torch.convert(model, sample_input)
    
    output_edge = edge_model(*sample_input)
    
    
    torch_out_np = output_torch.detach().numpy()
    edge_out_np = output_edge  # ya debería ser numpy
    print(np.allclose(torch_out_np, edge_out_np, atol=1e-5, rtol=1e-5))
    edge_model.export("./models/tflite/mobilevit_s_final.tflite")


