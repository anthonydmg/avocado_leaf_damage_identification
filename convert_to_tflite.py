import torch
import torch.nn as nn
from torchvision import models

import onnx
import tf2onnx
import tensorflow as tf
import numpy as np


# ================================================================
# 1. CARGAR TU MODELO (igual al que usaste entrenando)
# ================================================================
def load_model(num_classes, device):
    print("Cargando modelo EfficientNetV2-S pre-entrenado...")

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)

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
def export_to_onnx(model, save_path="model.onnx", input_size=(1,3,224,224)):
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
# ================================================================
def convert_onnx_to_tf(onnx_path="model.onnx", saved_model_dir="saved_model"):
    print("\n🔄 Convirtiendo ONNX → TensorFlow...")
    onnx_model = onnx.load(onnx_path)

    tf_rep, _ = tf2onnx.convert.from_onnx(onnx_model, output_path=None)

    tf.saved_model.save(tf_rep, saved_model_dir)
    print(f"✅ SavedModel generado en: {saved_model_dir}")


# ================================================================
# 4. TENSORFLOW → TFLITE
# ================================================================
def convert_tf_to_tflite(saved_model_dir="saved_model",
                         tflite_path="model.tflite",
                         quantize=False):

    print("\n📦 Convirtiendo a TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        print("   🔧 Aplicando cuantización INT8...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"🎉 TFLite generado: {tflite_path}")


# ================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================
if __name__ == "__main__":

    # ---- AJUSTAR ESTO ----
    num_classes = 3  # Cambia según tu dataset
    model_path = "efficientnetv2_s_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar modelo base
    model = load_model(num_classes, device)

    # 2. Cargar tus pesos entrenados
    print("\n📥 Cargando pesos entrenados...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("   ✔ Pesos cargados correctamente.")

    # 3. Exportar a ONNX
    export_to_onnx(model, "model.onnx")

    # 4. Convertir ONNX → TensorFlow
    convert_onnx_to_tf("model.onnx", "saved_model")

    # 5. Convertir a TFLite
    convert_tf_to_tflite("saved_model", "model_fp32.tflite", quantize=False)   # FP32
    convert_tf_to_tflite("saved_model", "model_int8.tflite", quantize=True)    # INT8

    print("\n🚀 Conversión completa con éxito.\n")
