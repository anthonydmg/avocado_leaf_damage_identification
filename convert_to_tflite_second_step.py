
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# ---------- 2. CONVERTIR ONNX → TF ----------
def convert_onnx_to_tf(onnx_path="model.onnx", tf_path="./saved_model"):
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    tf_rep.export_graph(tf_path)
    print(f"✔ SavedModel guardado en: {tf_path}")


# ---------- 3. CONVERTIR TF → TFLite ----------
def convert_tf_to_tflite(savedmodel_dir="./saved_model", output_tflite="model.tflite"):
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_tflite, "wb") as f:
        f.write(tflite_model)

    print(f"✔ Modelo TFLite guardado en: {output_tflite}")


# ---------- EJECUCIÓN ----------
if __name__ == "__main__":
    # Carga de tu modelo entrenado
    convert_onnx_to_tf("model.onnx", "./saved_model")
    #convert_tf_to_tflite("./saved_model", "final_model.tflite")
