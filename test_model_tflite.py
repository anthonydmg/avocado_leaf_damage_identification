import numpy as np
import tensorflow as tf
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="./models/tflite/efficientnetv2_s_final.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Entrada:", input_details)
print("Salida:", output_details)

img = Image.open("./datos/dataset_split/test/mosa-blanca-aleuropleurocelus-haz/20251107_113231.jpg").resize((224,224))
img = np.array(img, dtype=np.float32)
img = img / 255.0
img = img.transpose(2,0,1)
img = np.expand_dims(img, 0)
print("img:", img.shape)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicción:", output_data)