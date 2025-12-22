
import torch
import numpy as np
import torchvision
import ai_edge_torch

def main():
    # 1. Instanciar el modelo PyTorch (ejemplo: ResNet18 pre-entrenado)
    model = torchvision.models.resnet18(
        torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ).eval()

    # 2. Crear una entrada de ejemplo (sample input)
    #    Aquí usamos batch size = 1, 3 canales (RGB), tamaño 224x224
    sample_input = (torch.randn(1, 3, 224, 224),)

    # 3. Hacer una inferencia con PyTorch para comparar después
    torch_output = model(*sample_input)

    # 4. Convertir el modelo PyTorch a LiteRT usando ai-edge-torch
    edge_model = ai_edge_torch.convert(model, sample_input)

    # 5. Hacer inferencia con el modelo convertido
    edge_output = edge_model(*sample_input)

    # 6. Validar que las salidas sean similares
    #    Convertimos a NumPy para compararlas
    torch_out_np = torch_output.detach().numpy()
    edge_out_np = edge_output  # ya debería ser numpy

    if np.allclose(torch_out_np, edge_out_np, atol=1e-5, rtol=1e-5):
        print("✅ Inferencia OK: salidas similares entre PyTorch y TFLite (LiteRT)")
    else:
        print("❗ Las salidas difieren. Revisa el modelo o los parámetros de conversión.")
        print("Torch output:", torch_out_np)
        print("Edge output:", edge_out_np)

    # 7. Exportar el modelo convertido a un archivo .tflite
    tflite_filename = "resnet18_litert.tflite"
    edge_model.export(tflite_filename)
    print(f"Modelo exportado a {tflite_filename}")

if __name__ == "__main__":
    main()